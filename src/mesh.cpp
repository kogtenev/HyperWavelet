#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <numeric>
#include <queue>

#include "mesh.h"
#include "helpers.h"

using hyper_wavelet::SegmentTree;
using hyper_wavelet::CartesianToSphere;
using hyper_wavelet::DistanceToTrue;
using hyper_wavelet::Hedgehog;

namespace hyper_wavelet_2d {


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
    _fullArea = 0.;
    for (const auto& rectangle: _data) {
        _fullArea += rectangle.area;
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
    parts.resize(num_parts+1);
    for (const auto& edge: edges) {
        if (partition[edge.first] == partition[edge.second]) {
            parts[partition[edge.first]].push_back(edge);
        }
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

void RectangleMesh::_ReorientLocalBases(const int begin, const int end) {
    const int numPhi = 18, numTheta = 9;
    SegmentTree Phi(0, 2 * M_PI, numPhi);
    SegmentTree Theta(-M_PI_2, M_PI_2, numTheta);
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> directionIsUsed(numPhi, numTheta);
    directionIsUsed.fill(false);
    for (int i = begin; i < end; ++i) {
        double phi, theta;
        CartesianToSphere(_data[i].normal, phi, theta);
        int k = Phi.Find(phi);
        int l = Theta.Find(theta);
        directionIsUsed(k, l) = true;
        directionIsUsed(numPhi - k - 1, l) = true;
        if (l == 0 || l == numTheta - 1) {
            for (int j = 0; j < numPhi; ++j) {
                directionIsUsed(j, l) = true;
            }
        }
    }
    int maxDist = 0, phiOpt = -1, thetaOpt = -1;
    for (int i = 0; i < numPhi; ++i) {
        for (int j = 0; j < numTheta; ++j) {
            int newDist = DistanceToTrue(i, j, directionIsUsed);
            if (newDist > maxDist) {
                maxDist = newDist;
                phiOpt = i;
                thetaOpt = j;
            }
        }
    }
    if (maxDist == 0) {
        throw std::runtime_error("Cannot comb the hedgehog!");
    }
    Hedgehog hedgehog(Phi.Median(phiOpt), Theta.Median(thetaOpt));
    for (int i = begin; i < end; ++i) {
        _data[i].e1 = hedgehog.Comb(_data[i].normal);
        _data[i].e2 = _data[i].normal.cross(_data[i].e1);
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
    fout << "CELL_DATA " << ncells << '\n';
    fout << "VECTORS e1 double\n";
    for (int i = start; i < finish; ++i) {
        fout << rectangles[i].e1 << '\n';
    }
    fout << "VECTORS e2 double\n";
    for (int i = start; i < finish; ++i) {
        fout << rectangles[i].e2 << '\n';
    }
    fout << "VECTORS n double\n";
    for (int i = start; i < finish; ++i) {
        fout << rectangles[i].normal << '\n';
    }
    fout.close();
}

RectangleMesh::RectangleMesh(const double r): r(r) {
    std::cout << "Reading mesh" << std::endl;
    std::ifstream fin("input/main.txt");
    
    int nmodules;
    fin >> nmodules;

    std::cout << "Number of modules: " << nmodules << std::endl;

    _offsets.resize(nmodules+1);
    for (int m = 0; m < nmodules+1; ++m) {
        fin >> _offsets[m];
    }
    
    const int ncellsFull = _offsets.back(); 
    std::cout << "Full number of cells: " << ncellsFull << std::endl; 

    _moduleAreas.resize(nmodules);
    _graphEdges.resize(nmodules);
    _data.resize(ncellsFull);
    _fullArea = 0.;
    
    for (int m = 0; m < nmodules; ++m) {
        _ReadData(m); 
        _ReorientLocalBases(_offsets[m], _offsets[m+1]); 
    }

    std::cout << "Reading mesh done\n" << std::endl;
}

void RectangleMesh::FormWaveletMatrix() {
    std::cout << "Forming wavelet matrix" << std::endl;

    const int meshSize = _data.size();
    const int nmodules = _offsets.size() - 1;

    _wmatrix.starts.resize(meshSize);
    _wmatrix.medians.resize(meshSize);
    _wmatrix.ends.resize(meshSize);
    _wmatrix.rowLevels.resize(meshSize);
    _wmatrix.modules.resize(meshSize);
    _levels.resize(nmodules);
    
    for (int m = 0; m < nmodules; ++m) { 
        _FormWaveletSubmatrix(m); 
    }
    _PrepareSpheres();

    std::cout << "Wavelet matrix is ready" << std::endl;
    std::cout << "Max. number of levels: " << *std::max_element(_levels.begin(), _levels.end()) << '\n' << std::endl;
}

void RectangleMesh::_PrepareSpheres() {
    int nrows = _data.size();
    _wmatrix.spheres.resize(nrows);
    for (int row = 0; row < nrows; ++row) {
        Eigen::Vector3d center;
        center.fill(0.);
        int n = _wmatrix.ends[row] - _wmatrix.starts[row];
        for (int i = _wmatrix.starts[row]; i < _wmatrix.ends[row]; ++i) {
            center += _data[i].center / n;
        }
        double radious = 0.;
        for (int i = _wmatrix.starts[row]; i < _wmatrix.ends[row]; ++i) {
            double new_radious = (_data[i].center - center).norm() + _data[i].diameter;
            if (new_radious > radious) {
                radious = new_radious;
            }
        }
        radious *= r;
        _wmatrix.spheres[row] = {radious, center};
    }
}

void RectangleMesh::_ReadData(const int module) {
    std::ifstream fin("input/mesh" + std::to_string(module) + ".ply", std::ios::in);

    int npoints = ParseIntFromString(fin, "element vertex ");
    int ncells  = ParseIntFromString(fin, "element face ");
    SkipHeader(fin);

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

    for (int i = 0; i < 4*ncells; i += 4) {
        Eigen::Vector3d a;
        a << coords[3*faces[i]],   coords[3*faces[i] + 1],   coords[3*faces[i] + 2];

        Eigen::Vector3d b;
        b << coords[3*faces[i+1]], coords[3*faces[i+1] + 1], coords[3*faces[i+1] + 2];

        Eigen::Vector3d c;
        c << coords[3*faces[i+2]], coords[3*faces[i+2] + 1], coords[3*faces[i+2] + 2];

        Eigen::Vector3d d;
        d << coords[3*faces[i+3]], coords[3*faces[i+3] + 1], coords[3*faces[i+3] + 2];

        _data[_offsets[module] + i / 4] = Rectangle(a, b, c, d);
    }

    _moduleAreas[module] = 0.;
    for (int i = _offsets[module]; i < _offsets[module + 1]; ++i) {
        _moduleAreas[module] += _data[i].area;
    }
    _fullArea += _moduleAreas[module];

    std::string buffer;
    fin.open("input/graph" + std::to_string(module) + ".txt", std::ios::in);
    _graphEdges[module].reserve(8 * ncells);
    while (std::getline(fin, buffer)) {
        std::stringstream stream(buffer);
        int i, j;
        stream >> i >> j;
        _graphEdges[module].push_back({i, j});
    }
    std::vector<std::pair<int, int>> edges_reversed(_graphEdges[module].size());
    for (int i = 0; i < edges_reversed.size(); ++i) {
        edges_reversed[i] = {_graphEdges[module][i].second, _graphEdges[module][i].first};
    }
    _graphEdges[module].insert(_graphEdges[module].end(), edges_reversed.begin(), edges_reversed.end());
    std::sort(_graphEdges[module].begin(), _graphEdges[module].end(),
        [](const auto& a, const auto& b){ return a.first < b.first; });
}

void RectangleMesh::_FormWaveletSubmatrix(const int module) {
    int nvertices = _offsets[module + 1] - _offsets[module]; 
    int row = _offsets[module];

    _wmatrix.starts[row] = _offsets[module];
    _wmatrix.medians[row] = _offsets[module + 1]; 
    _wmatrix.ends[row] = _offsets[module + 1]; 
    _wmatrix.rowLevels[row] = 0;

    ++row;

    std::vector<int> barriers = {0, nvertices};
    std::vector<std::vector<std::pair<int, int>>> subgraphs_edges = {std::move(_graphEdges[module])};
    int diameter = nvertices;
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

            ReorderMesh(_offsets[module] + barriers[i], partition, _data);

            new_barriers.push_back(left_pivoting.size());
            new_barriers.push_back(right_pivoting.size());

            new_subgraph_edges.push_back(std::move(left_edges));
            new_subgraph_edges.push_back(std::move(right_edges));

            _wmatrix.starts[row] = _offsets[module] + barriers[i];
            _wmatrix.medians[row] = _offsets[module] + barriers[i] + left_pivoting.size();
            _wmatrix.ends[row] = _offsets[module] + barriers[i+1];
            _wmatrix.rowLevels[row] = level;
            _wmatrix.modules[row] = module;
            
            ++row;
        }
        std::partial_sum(new_barriers.begin(), new_barriers.end(), new_barriers.begin());
        barriers = std::move(new_barriers);

        subgraphs_edges = std::move(new_subgraph_edges);
        diameter = GetDiameter(barriers);

        ++level; 
    }

    if (row != _offsets[module + 1]) {
        throw std::runtime_error("Cannot prepare wavelet matrix!");
    }

    _levels[module] = level - 1;

    PrintSubmesh(_data, _offsets[module], _offsets[module+1], "module" + std::to_string(module) + ".vtk");
}

}