#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <numeric>

#include "mesh.h"
#include "helpers.h"

namespace hyper_wavelet_2d {

// By value, not by reference!
inline bool LinearDependent(Eigen::Vector3d x, Eigen::Vector3d y, Eigen::Vector3d z, double tol = 1e-12) {
    double zNorm = z.norm();
    x /= x.norm();
    y = y - y.dot(x) * x;
    y /= y.norm();
    z = z - z.dot(x) * x;
    z = z - z.dot(y) * y;
    return z.norm() < zNorm * tol;
}

RectangleMesh::RectangleMesh(int nx, int ny, 
    const std::function<Eigen::Vector3d(double, double)>& surfaceMap): 
                                    _nx(nx), _ny(ny), surfaceMap(surfaceMap) {

    double hx = 1. / nx, hy = 1. / ny;
    _data.resize(nx * ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            const Eigen::Vector3d a = surfaceMap(i * hx, j * hy);
            const Eigen::Vector3d b = surfaceMap((i + 1) * hx, j * hy);
            const Eigen::Vector3d c = surfaceMap((i + 1) * hx, (j + 1) * hy);
            const Eigen::Vector3d d = surfaceMap(i * hx, (j + 1) * hy);
            _data[nx * j + i] = Rectangle(a, b, c, d);
        }
    }
    _area = 0.;
    for (const auto& rectangle: _data) {
        _area += rectangle.area;
    }
}

int ParseIntFromString(std::ifstream& fin, const std::string& prefix) {
    int result;
    bool found = false;
    std::string buffer;
    while (std::getline(fin, buffer)) {
        if (buffer.find(prefix) != std::string::npos) {
            std::stringstream stream(buffer);
            stream.ignore(prefix.size());
            stream >> result;
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::runtime_error("Cannot parse mesh file!");
    }
    return result;
}

void SkipHeader(std::ifstream& fin, const std::string& stopLine = "end_header") {
    bool success = false;
    std::string buffer;
    while (std::getline(fin, buffer)) {
        if (buffer == stopLine) {
            success = true;
            break;
        }
    }
    if (!success) {
        throw std::runtime_error("Cannot skip header!");
    }
}

void GetPivoting(
    const std::vector<idx_t>& partition, 
    std::map<int, int>& left_pivoting,
    std::map<int, int>& right_pivoting
) {    
    std::vector<int> left_piv_inverse, right_piv_inverse;
    for (int i = 0; i < partition.size(); ++i) {
        if (partition[i] == 0) {
            left_piv_inverse.push_back(i);
        } else {
            right_piv_inverse.push_back(i);
        }
    }
    for (int i = 0; i < left_piv_inverse.size(); ++i) {
        left_pivoting[left_piv_inverse[i]] = i;
    }
    for (int i = 0; i < right_piv_inverse.size(); ++i) {
        right_pivoting[right_piv_inverse[i]] = i;
    }
}

void GraphEdgesToCsr(
    const std::vector<std::pair<int, int>>& edges,
    std::vector<idx_t>& csr_starts, 
    std::vector<idx_t>& csr_list
) {
    csr_list.reserve(edges.size());
    for (const auto& edge: edges) {
        ++csr_starts[edge.first + 1];
        csr_list.push_back(edge.second);
    }
    std::partial_sum(csr_starts.begin(), csr_starts.end(), csr_starts.begin());
}

void SplitEdges(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<idx_t>& partition,
    std::vector<std::pair<int, int>>& left_part,
    std::vector<std::pair<int, int>>& right_part
) {
    for (const auto& edge: edges) {
        if (partition[edge.first] == 0 and partition[edge.second] == 0) {
            left_part.push_back(edge);
        } else if (partition[edge.first] == 1 and partition[edge.second] == 1) {
            right_part.push_back(edge);
        }
    }
}

void SplitEdges(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<idx_t>& partition,
    std::vector<std::vector<std::pair<int, int>>>& parts
) {
    const int num_parts = *std::max_element(partition.begin(), partition.end());
    parts.resize(num_parts);
    for (const auto& edge: edges) {
        if (partition[edge.first] == partition[edge.second]) {
            parts[partition[edge.first]].push_back(edge);
        }
    }
}

void BreadthFirstSearch(
    const int start_vertex,
    const int visit_marker,
    const std::vector<idx_t>& csr_starts,
    const std::vector<idx_t>& csr_list, 
    std::vector<idx_t>& visited,
    int& not_visited
) {
    not_visited = 0;
    std::vector<idx_t> active_set = {start_vertex};
    while (active_set.size()) {
        std::vector<idx_t> new_set;
        for (int vertex: active_set) {
            visited[vertex] = visit_marker;
            if (not_visited == vertex) {
                ++not_visited;
            }
            for (int j = csr_starts[vertex]; j < csr_starts[vertex+1]; ++j) {
                if (!visited[csr_list[j]]) {
                    new_set.push_back(csr_list[j]);
                }
            }
            active_set = std::move(new_set);
        }
    } 
}

void GetConnectedComponents(
    const int nvertices,
    const std::vector<std::pair<int, int>>& edges,
    std::vector<idx_t>& partition,
    std::vector<std::vector<std::pair<int, int>>>& parts
) {
    std::vector<idx_t> csr_starts(nvertices, 0), csr_list;
    GraphEdgesToCsr(edges, csr_starts, csr_list);
    int start_vertex = 0, visit_marker = 1;
    partition.resize(nvertices, 0);
    while (start_vertex < nvertices) {
        int not_visited;
        BreadthFirstSearch(start_vertex, visit_marker, csr_starts, csr_list, partition, not_visited);
        start_vertex = not_visited;
        ++visit_marker;
    }   
}

void RenumberVertices(
    std::vector<std::pair<int, int>>& edges,
    const std::map<int, int>& mesh_pivoting
) {    
    for (int i = 0; i < edges.size(); ++i) {
        edges[i].first  = mesh_pivoting.at(edges[i].first);
        edges[i].second = mesh_pivoting.at(edges[i].second);
    }
}

void ReorderMesh(
    int start,
    const std::vector<idx_t>& partition, 
    std::vector<Rectangle>& rectangles
) {
    Rectangle* const data = &rectangles[start];
    std::vector<Rectangle> left_buffer, right_buffer;
    for (int i = 0; i < partition.size(); ++i) {
        if (partition[i] == 0) {
            left_buffer.push_back(data[i]);
        } else {
            right_buffer.push_back(data[i]);
        }
    }
    int i = 0;
    for (auto& rect: left_buffer) {
        data[i] = rect;
        ++i;
    }
    for (auto& rect: right_buffer) {
        data[i] = rect;
        ++i;
    } 
}

int GetDiameter(const std::vector<int>& barriers) {
    int result = 0;
    for (int i = 0; i < barriers.size() - 1; ++i) {
        if (barriers[i+1] - barriers[i] > result) {
            result = barriers[i+1] - barriers[i];
        }
    }
    return result;
}

void Call_METIS(
    const std::vector<idx_t>& csr_starts, 
    const std::vector<idx_t>& csr_list, 
    std::vector<idx_t>& partition
) {
    idx_t nvtxs = csr_starts.size() - 1, ncon = 1, nparts = 2, objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    int errcode = METIS_PartGraphKway(&nvtxs, &ncon, (idx_t*)csr_starts.data(), (idx_t*)csr_list.data(), 
                        NULL, NULL, NULL, &nparts, NULL, NULL, options, &objval, partition.data());

    switch (errcode) {
        case METIS_OK: break;
        case METIS_ERROR_INPUT:
            std::cout << "METIS error input!\n";
            break;
        case METIS_ERROR_MEMORY:
            std::cout << "METIS error memory!\n";
            break;
        case METIS_ERROR:
            std::cout << "Unknown METIS error!\n";
            break;
        default:
            std::cout << "Strange METIS output!\n";
    }
}

void PrintSubmesh(const std::vector<Rectangle>& rectangles, 
        int start, int finish, const std::string& fileName) {

    std::ofstream fout(fileName, std::ios::out);
    fout << "# vtk DataFile Version 3.0\n";
    fout << "Wavelet submesh\n";
    fout << "ASCII\n";
    fout << "DATASET POLYDATA\n";
    const int npoints = (finish - start) * 4;
    const int ncells  = finish - start;
    fout << "POINTS " << npoints << " double\n";
    for (int i = start; i < finish; ++i) {
        fout << rectangles[i].a << '\n' << rectangles[i].b << '\n' 
             << rectangles[i].c << '\n' << rectangles[i].d << '\n';
    }
    fout << "POLYGONS " << ncells << ' ' << 5 * ncells << '\n';
    for (int i = 0; i < npoints; i += 4) {
        fout << "4 " << i << ' ' << i+1 << ' ' << i+2 << ' ' << i+3 << '\n';
    }
    fout.close();
}

RectangleMesh::RectangleMesh(
    const double r, 
    const std::string& meshFile, 
    const std::string& graphFile
): r(r) 
{
    std::ifstream fin(meshFile, std::ios::in);

    int npoints = ParseIntFromString(fin, "element vertex ");
    int ncells  = ParseIntFromString(fin, "element face ");
    SkipHeader(fin);

    std::cout << "\nReading mesh\n";
    std::cout << "Number of vertices: " << npoints << '\n';
    std::cout << "Number of faces: " << ncells << '\n';

    std::vector<double> coords(3 * npoints);
    for (int i = 0; i < npoints; ++i) {
        fin >> coords[3 * i];
        fin >> coords[3 * i + 1];
        fin >> coords[3 * i + 2];
    }

    std::vector<int> faces(4 * ncells);
    for (int i = 0; i < ncells; ++i) {
        fin.ignore(256, ' ');
        fin >> faces[4 * i];
        fin >> faces[4 * i + 1];
        fin >> faces[4 * i + 2];
        fin >> faces[4 * i + 3];
    }
    fin.close();

    _data.resize(ncells);
    for (int i = 0; i < 4*ncells; i += 4) {
        Eigen::Vector3d a;
        a << coords[3*faces[i]],   coords[3*faces[i] + 1],   coords[3*faces[i] + 2];

        Eigen::Vector3d b;
        b << coords[3*faces[i+1]], coords[3*faces[i+1] + 1], coords[3*faces[i+1] + 2];

        Eigen::Vector3d c;
        c << coords[3*faces[i+2]], coords[3*faces[i+2] + 1], coords[3*faces[i+2] + 2];

        Eigen::Vector3d d;
        d << coords[3*faces[i+3]], coords[3*faces[i+3] + 1], coords[3*faces[i+3] + 2];

        _data[i / 4] = Rectangle(a, b, c, d);
    }
    std::cout << "Mesh is ready\n";

    _area = 0.;
    for (const auto& rectangle: _data) {
        _area += rectangle.area;
    }

    std::cout << "Reading mesh dual graph\n";
    std::string buffer;
    fin.open(graphFile, std::ios::in);
    while (std::getline(fin, buffer)) {
        std::stringstream stream(buffer);
        int i, j;
        stream >> i >> j;
        _graphEdges.push_back({i, j});
    }
    std::vector<std::pair<int, int>> edges_reversed(_graphEdges.size());
    for (int i = 0; i < edges_reversed.size(); ++i) {
        edges_reversed[i] = {_graphEdges[i].second, _graphEdges[i].first};
    }
    _graphEdges.insert(_graphEdges.end(), edges_reversed.begin(), edges_reversed.end());
    std::sort(_graphEdges.begin(), _graphEdges.end(),
        [](const auto& a, const auto& b){ return a.first < b.first; });
    std::cout << "Mesh graph is ready\n\n";
}

void RectangleMesh::FormWaveletMatrix() {
    std::cout << "Forming wavelet matrix" << std::endl;

    int nvertices = _data.size();
    int diameter = nvertices;

    _wmatrix.starts.resize(nvertices);
    _wmatrix.medians.resize(nvertices);
    _wmatrix.ends.resize(nvertices);
    _wmatrix.rowLevels.resize(nvertices);

    _wmatrix.starts[0] = 0;
    _wmatrix.medians[0] = nvertices;
    _wmatrix.ends[0] = nvertices;
    _wmatrix.rowLevels[0] = 0;

    std::vector<int> barriers = {0, nvertices};
    std::vector<std::vector<std::pair<int, int>>> subgraphs_edges = {std::move(_graphEdges)};
    int row = 1;
    int level = 1;

    while (diameter > 1) {
        std::vector<int> new_barriers = {0};
        std::vector<std::vector<std::pair<int, int>>> new_subgraph_edges;

        for (int i = 0; i < subgraphs_edges.size(); ++i) {
            nvertices = barriers[i+1] - barriers[i];

            if (nvertices < 2) {
                new_barriers.push_back(nvertices);
                new_subgraph_edges.push_back(std::move(subgraphs_edges[i]));
                continue;
            }
            
            std::vector<idx_t> csr_starts(nvertices+1, 0), csr_list;

            GraphEdgesToCsr(subgraphs_edges[i], csr_starts, csr_list);
            std::vector<idx_t> partition(nvertices); 

            Call_METIS(csr_starts, csr_list, partition); 
            
            std::map<int, int> left_pivoting, right_pivoting;
            GetPivoting(partition, left_pivoting, right_pivoting);

            std::vector<std::pair<int, int>> left_edges, right_edges;
            SplitEdges(subgraphs_edges[i], partition, left_edges, right_edges);
            
            RenumberVertices(left_edges, left_pivoting);
            RenumberVertices(right_edges, right_pivoting);

            ReorderMesh(barriers[i], partition, _data);

            new_barriers.push_back(left_pivoting.size());
            new_barriers.push_back(right_pivoting.size());

            new_subgraph_edges.push_back(std::move(left_edges));
            new_subgraph_edges.push_back(std::move(right_edges));

            _wmatrix.starts[row] = barriers[i];
            _wmatrix.medians[row] = barriers[i] + left_pivoting.size();
            _wmatrix.ends[row] = barriers[i+1];
            _wmatrix.rowLevels[row] = level;
            
            ++row;
        }
        std::partial_sum(new_barriers.begin(), new_barriers.end(), new_barriers.begin());
        barriers = std::move(new_barriers);

        subgraphs_edges = std::move(new_subgraph_edges);
        diameter = GetDiameter(barriers);

        ++level; 
    }

    if (row != _data.size()) {
        throw std::runtime_error("Cannot prepare wavelet matrix!");
    }

    _PrepareSpheres();
    _levels = level - 1;

    std::cout << "Wavelet matrix is ready" << std::endl;
    std::cout << "Number of levels: " << level - 1 << '\n' << std::endl;

    PrintSubmesh(_data, _wmatrix.starts[0],  _wmatrix.ends[0],  "submesh0.vtk");
    PrintSubmesh(_data, _wmatrix.starts[2],  _wmatrix.ends[2],  "submesh2.vtk");
    PrintSubmesh(_data, _wmatrix.starts[8],  _wmatrix.ends[8],  "submesh8.vtk");
    PrintSubmesh(_data, _wmatrix.starts[49], _wmatrix.ends[49], "submesh49.vtk");
}

void RectangleMesh::PrintLocalBases() const {
std::ofstream fout("local_bases.vtk", std::ios::out);
    fout << "# vtk DataFile Version 3.0\n";
    fout << "Surface electric current\n";
    fout << "ASCII\n";
    fout << "DATASET POLYDATA\n";
    const int npoints = Data().size() * 4;
    const int ncells  = Data().size();
    fout << "POINTS " << npoints << " double\n";
    for (const auto& rectangle: Data()) {
        fout << rectangle.a << '\n' << rectangle.b << '\n' << rectangle.c << '\n' << rectangle.d << '\n';
    }
    int i = 0;
    fout << "POLYGONS " << ncells << ' ' << 5 * ncells << '\n';
    for (const auto& rectangle: Data()) {
        fout << "4 " << i << ' ' << i+1 << ' ' << i+2 << ' ' << i+3 << '\n';
        i += 4; 
    }
    fout << "CELL_DATA " << ncells << '\n';
    fout << "VECTORS e1 double\n";
    for (const auto& rectangle: Data()) {
        fout << rectangle.e1 << '\n';
    }
    fout << "VECTORS e2 double\n";
    for (const auto& rectangle: Data()) {
        fout << rectangle.e2 << '\n';
    }
    fout << "VECTORS n double\n";
    for (const auto& rectangle: Data()) {
        fout << rectangle.normal << '\n';
    }
    fout.close();
}

}