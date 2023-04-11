#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <numeric>

#include "metis.h"

#include "surface.h"
#include "helpers.h"
#include "Haar.h"
#include "bases.h"

using complex = std::complex<double>;
using hyper_wavelet::Profiler;
using hyper_wavelet::Interval;
using hyper_wavelet::Distance;

using namespace std::complex_literals;

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

void GraphEdgesToCsr(
    const std::vector<std::pair<int, int>>& edges,
    std::vector<idx_t>& csr_starts, 
    std::vector<idx_t>& csr_list) {

    csr_list.reserve(edges.size());
    for (const auto& edge: edges) {
        ++csr_starts[edge.first + 1];
        csr_list.push_back(edge.second);
    }
    std::partial_sum(csr_starts.begin(), csr_starts.end(), csr_starts.begin());
}

void GetPivoting(
    const std::vector<idx_t>& partition, 
    std::map<int, int>& left_pivoting,
    std::map<int, int>& right_pivoting) {
    
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

void SplitEdges(
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<idx_t>& partition,
    std::vector<std::pair<int, int>>& left_part,
    std::vector<std::pair<int, int>>& right_part) {

    for (const auto& edge: edges) {
        if (partition[edge.first] == 0 and partition[edge.second] == 0) {
            left_part.push_back(edge);
        } else if (partition[edge.first] == 1 and partition[edge.second] == 1) {
            right_part.push_back(edge);
        }
    }
}

void RenumberVertices(
    std::vector<std::pair<int, int>>& edges,
    const std::map<int, int>& mesh_pivoting) {
    
    for (int i = 0; i < edges.size(); ++i) {
        edges[i].first = mesh_pivoting.at(edges[i].first);
        edges[i].second = mesh_pivoting.at(edges[i].second);
    }
}

void ReorderMesh(
    int start,
    const std::vector<idx_t>& partition, 
    std::vector<Rectangle>& rectangles) {

    Rectangle* const data = &rectangles[start];
    std::vector<Rectangle> left_buffer, right_buffer;
    for (int i = 0; i < partition.size(); ++i) {
        if (partition[i] == 0) {
            left_buffer.push_back(std::move(data[i]));
        } else {
            right_buffer.push_back(std::move(data[i]));
        }
    }
    int i = 0;
    for (auto& rect: left_buffer) {
        data[i] = std::move(rect);
        ++i;
    }
    for (auto& rect: right_buffer) {
        data[i] = std::move(rect);
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
    std::vector<idx_t>& partition) {

    idx_t nvtxs = csr_starts.size() - 1, ncon = 1, nparts = 2, objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    int errcode = METIS_PartGraphRecursive(&nvtxs, &ncon, (idx_t*)csr_starts.data(), (idx_t*)csr_list.data(), 
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

RectangleMesh::RectangleMesh(const std::string& meshFile, const std::string& graphFile) {
    std::ifstream fin(meshFile, std::ios::in);

    int npoints = ParseIntFromString(fin, "element vertex ");
    int ncells  = ParseIntFromString(fin, "element face ");
    SkipHeader(fin);

    std::cout << "Reading mesh\n";
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
    std::cout << "Mesh is written\n";

    if (graphFile.size()) {
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
}

void RectangleMesh::FormWaveletMatrix() {
    int nvertices = _data.size();
    int diameter = nvertices;

    _wmatrix.starts.resize(nvertices);
    _wmatrix.medians.resize(nvertices);
    _wmatrix.ends.resize(nvertices);

    _wmatrix.starts[0] = 0;
    _wmatrix.medians[0] = nvertices;
    _wmatrix.ends[0] = nvertices;

    std::vector<int> barriers = {0, nvertices};
    std::vector<std::vector<std::pair<int, int>>> subgraphs_edges = {std::move(_graphEdges)};
    int row = 1;

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
            
            ++row;
        }
        std::partial_sum(new_barriers.begin(), new_barriers.end(), new_barriers.begin());
        barriers = std::move(new_barriers);

        subgraphs_edges = std::move(new_subgraph_edges);
        diameter = GetDiameter(barriers); 
    }

    if (row != _data.size()) {
        throw std::runtime_error("Cannot prepare wavelet matrix!");
    }

    PrintSubmesh(_data, _wmatrix.starts[0],  _wmatrix.ends[0],  "submesh0.vtk");
    PrintSubmesh(_data, _wmatrix.starts[2],  _wmatrix.ends[2],  "submesh2.vtk");
    PrintSubmesh(_data, _wmatrix.starts[3],  _wmatrix.ends[3],  "submesh3.vtk");
    PrintSubmesh(_data, _wmatrix.starts[8],  _wmatrix.ends[8],  "submesh8.vtk");
    PrintSubmesh(_data, _wmatrix.starts[15], _wmatrix.ends[15], "submesh15.vtk");
    PrintSubmesh(_data, _wmatrix.starts[32], _wmatrix.ends[32], "submesh32.vtk");
    PrintSubmesh(_data, _wmatrix.starts[49], _wmatrix.ends[49], "submesh57.vtk");
}

void PrepareSupports1D(std::vector<Interval>& intervals, int n) {
    intervals.resize(n);
    intervals[0] = {0., 1.};
    intervals[1] = {0., 1.};

    int numOfSupports = 2;
    double h = 0.5;
    int cnt = 2;

    while (numOfSupports < n) {
        for (int i = 0; i < numOfSupports; i++) {
            intervals[cnt] = {i * h, (i + 1) * h};
            cnt++; 
        }
        h /= 2;
        numOfSupports *= 2;
    }

    if (cnt != n) {
       throw std::runtime_error("Cannot prepare 1D Haar supports!");
    }
}

std::array<Rectangle, 4> Bisection(const Rectangle& A) {
    std::array<Rectangle, 4> result;
    result[0] = Rectangle(A.a, (A.a + A.b) / 2, A.center, (A.a + A.d) / 2);
    result[1] = Rectangle((A.a + A.b) / 2, A.b, (A.b + A.c) / 2, A.center);
    result[2] = Rectangle(A.center, (A.b + A.c) / 2, A.c, (A.c + A.d) / 2);
    result[3] = Rectangle((A.a + A.d) / 2, A.center, (A.c + A.d) / 2, A.d);
    return result;
}

std::vector<Rectangle> RefineRectangle(const Rectangle& rectangle, int numLevels) {
    std::vector<Rectangle> result {rectangle};
    std::vector<Rectangle> helper;
    for (int level = 0; level < numLevels; ++level) {
        helper.resize(0);
        for (const auto& rect: result) {
            auto refined = Bisection(rect);
            helper.insert(helper.end(), refined.begin(), refined.end());
        }
        result = std::move(helper);
    }
    return result;
}

void RectangleMesh::HaarTransform() {
    std::cout << "Preparing mesh for Haar basis\n";

    Profiler profiler;

    std::vector<Interval> supports_x, supports_y;
    PrepareSupports1D(supports_x, _nx);
    PrepareSupports1D(supports_y, _ny);

    for (int j = 0; j < _ny; j++) {
        for (int i = 0; i < _nx; i++) {
            const double a_x = supports_x[i].a;
            const double b_x = supports_x[i].b;
            const double a_y = supports_y[j].a;
            const double b_y = supports_y[j].b;

            Eigen::Vector3d r1 = surfaceMap(a_x, a_y);
            Eigen::Vector3d r2 = surfaceMap(b_x, a_y);
            Eigen::Vector3d r3 = surfaceMap(b_x, b_y);
            Eigen::Vector3d r4 = surfaceMap(a_x, b_y);

            _data[_nx * j + i] = Rectangle(r1, r2, r3, r4);
        }
    }
        
    std::cout << "Time for preparation: " << profiler.Toc() << " s.\n\n";
}

const std::function<Eigen::Vector3d(double, double)> _unitMap = [](double x, double y) {
    Eigen::Vector3d result;
    result[0] = x;
    result[1] = y;
    result[2] = 0.;
    return result;
};

inline double RectangleSurfaceSolver::_Smooth(double r) const {
    return r < _eps ? 3*r*r*r/_eps/_eps/_eps - 2*r*r/_eps/_eps : 1.;
}

Eigen::Vector3cd RectangleSurfaceSolver::
_MainKernelPart(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& x) {
    
    Eigen::Vector3cd AM = (x - a).cast<complex>();
    Eigen::Vector3cd BM = (x - b).cast<complex>();

    return (AM / AM.norm() + BM / BM.norm()) * (b - a).norm() / (AM.norm() * BM.norm() + AM.dot(BM));
}

inline Eigen::Vector3cd K1(const Eigen::Vector3d& j, const Eigen::Vector3d& x, const Eigen::Vector3d& y, double k) {
    const double R = (x - y).norm();
    const Eigen::Vector3d& r = (x - y) / R;
    std::complex<double> i = {0, 1};
    return (1. - std::exp(i*k*R) + i*k*R * std::exp(i*k*R)) / R / R / R * 
           (j.cast<complex>() - 3. * r.dot(j) * r.cast<complex>()) +
            k * k * std::exp(i*k*R) / R * (j.cast<complex>() - r.dot(j) * r.cast<complex>());
}

Eigen::Vector3cd RectangleSurfaceSolver::
_RegularKernelPart(const Eigen::Vector3d& J, const Rectangle& X, const Rectangle& X0) {
    const Eigen::Vector3d& x0 = X0.center;
    int levels = (x0 - X.center).norm() / std::sqrt(std::min(X.area, X0.area)) < _adaptation ? _refineLevels : 1;
    Eigen::Vector3cd result;
    result.fill({0., 0.});
    const auto& rectangles = RefineRectangle(X, levels);
    for (const auto& s: rectangles) {
        result += _Smooth((s.center - x0).norm()) * s.area * K1(J, x0, s.center, _k);
    }
    return result;
}

Eigen::Matrix2cd RectangleSurfaceSolver::_LocalMatrix(const Rectangle& X, const Rectangle& X0) {
    Eigen::Matrix2cd a;

    const Eigen::Vector3cd I_ab = _MainKernelPart(X.a, X.b, X0.center);
    const Eigen::Vector3cd I_bc = _MainKernelPart(X.b, X.c, X0.center);
    const Eigen::Vector3cd I_cd = _MainKernelPart(X.c, X.d, X0.center);
    const Eigen::Vector3cd I_da = _MainKernelPart(X.d, X.a, X0.center);

    double scale;
    Eigen::Vector3d t_ab = (X.b - X.a); scale = t_ab.norm();
    t_ab /= (scale > 0.) ? scale : 1;

    Eigen::Vector3d t_bc = (X.c - X.b); scale = t_bc.norm();
    t_bc /= (scale > 0.) ? scale : 1; 

    Eigen::Vector3d t_cd = (X.d - X.c); scale = t_cd.norm(); 
    t_cd /= (scale > 0.) ? scale : 1; 

    Eigen::Vector3d t_da = (X.a - X.d); scale = t_da.norm();
    t_da /= (scale > 0.) ? scale : 1;    

    Eigen::Vector3cd Ke1 = X.e1.cross(X.normal).dot(t_ab) * I_ab;
    Ke1 += X.e1.cross(X.normal).dot(t_bc) * I_bc;
    Ke1 += X.e1.cross(X.normal).dot(t_cd) * I_cd;
    Ke1 += X.e1.cross(X.normal).dot(t_da) * I_da;

    Eigen::Vector3cd Ke2 = X.e2.cross(X.normal).dot(t_ab) * I_ab;
    Ke2 += X.e2.cross(X.normal).dot(t_bc) * I_bc;
    Ke2 += X.e2.cross(X.normal).dot(t_cd) * I_cd;
    Ke2 += X.e2.cross(X.normal).dot(t_da) * I_da;

    Ke1 += _RegularKernelPart(X.e1, X, X0);
    Ke2 += _RegularKernelPart(X.e2, X, X0);

    a(0, 0) =  X0.normal.cast<complex>().cross(Ke1).dot(X0.e2.cast<complex>());
    a(1, 0) = -X0.normal.cast<complex>().cross(Ke1).dot(X0.e1.cast<complex>());
    a(0, 1) =  X0.normal.cast<complex>().cross(Ke2).dot(X0.e2.cast<complex>());
    a(1, 1) = -X0.normal.cast<complex>().cross(Ke2).dot(X0.e1.cast<complex>());

    return a;
}

void RectangleSurfaceSolver::_formBlockCol(Eigen::MatrixXcd& blockCol, int j) {
    blockCol.resize(_dim, 2);
    const auto& rectangles = _mesh.Data();
    #pragma omp parallel for
    for (int i = 0; i < _dim / 2; i++) {
        blockCol.block<2, 2>(2*i, 0) = _LocalMatrix(rectangles[j], rectangles[i]);
    }

    auto V0 = blockCol.col(0);
    auto V1 = blockCol.col(1);

    Subvector2D V0_x(V0, _dim / 2, 0);
    Subvector2D V0_y(V0, _dim / 2, 1);
    Subvector2D V1_x(V1, _dim / 2, 0);
    Subvector2D V1_y(V1, _dim / 2, 1);

    Haar2D(V0_x, _ny, _nx);
    Haar2D(V0_y, _ny, _nx);
    Haar2D(V1_x, _ny, _nx);
    Haar2D(V1_y, _ny, _nx);
}

RectangleSurfaceSolver::RectangleSurfaceSolver(int nx, int ny, double k, 
    const std::function<Eigen::Vector3d(double, double)>& surfaceMap
): _mesh(nx, ny, surfaceMap), _unitMesh(nx, ny, _unitMap), 
   _k(k), _nx(nx), _ny(ny), _dim(2*nx*ny), _eps(1./std::sqrt(nx*ny)/4),
   _adaptation(std::log2(1.*nx*ny)) {

}

RectangleSurfaceSolver::RectangleSurfaceSolver(double k): 
    _k(k), _nx(-1), _ny(-1), _eps(1e-3), _adaptation(1e-2) {}

void RectangleSurfaceSolver::FormFullMatrix() {
    std::cout << "Forming full matrix" << std::endl;
    std::cout << "Matrix size: " << _dim << " x " << _dim << std::endl;
    Profiler profiler;
    _fullMatrix.resize(_dim, _dim);
    const auto& rectangles = _mesh.Data();
    const int n = rectangles.size();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            _fullMatrix.block<2, 2>(2*i, 2*j) = _LocalMatrix(rectangles[j], rectangles[i]);
        }
    }
    std::cout << "Matrix is formed" << std::endl;
    std::cout << "Time for forming matrix: " << profiler.Toc() << " s." << std::endl << std::endl; 
}

void RectangleSurfaceSolver::FormTruncatedMatrix(double threshold, bool print) {
    RectangleMesh haarMesh = _unitMesh;
    haarMesh.HaarTransform();
    std::cout << "Forming truncated matrix\n";
    Profiler profiler;
    const auto& rectangles = haarMesh.Data();
    const int n = rectangles.size();
    std::vector<Eigen::Triplet<complex>> triplets;
    _truncMatrix.resize(_dim, _dim);
    _truncMatrix.makeCompressed();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (PlaneParRectDist(rectangles[j], rectangles[i]) < threshold) {
                triplets.push_back({2*i, 2*j, _fullMatrix(2*i, 2*j)});
                triplets.push_back({2*i+1, 2*j, _fullMatrix(2*i+1, 2*j)});
                triplets.push_back({2*i, 2*j+1, _fullMatrix(2*i, 2*j+1)});
                triplets.push_back({2*i+1, 2*j+1, _fullMatrix(2*i+1, 2*j+1)});
            }
        }
    }
    _truncMatrix.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << "Time for forming truncated matrix: " << profiler.Toc() << " s.\n"; 
    std::cout << "Proportion of nonzeros: " << 1. * triplets.size() / triplets.size() << "\n";

    if (print) {
        std::ofstream fout("trunc_mat.txt", std::ios::out);
        std::cout << "Printing truncated matrix" << '\n';
        for (const auto& triplet: triplets) {
            fout << triplet.col() << ' ' << triplet.row()
                 << ' ' << std::abs(triplet.value()) << '\n';
        }
        fout.close();    
    }
    std::cout << '\n';
}

void MakeHaarMatrix1D(int n, Eigen::MatrixXd& H) {
    H.resize(n, n);
    H.fill(0.);
    for (int i = 0; i < n; i++) {
        H(i, i) = 1.;
    }
    for (int j = 0; j < n; j++) {
        auto col = H.col(j);
        Haar(col);
    }
}

void MakeHaarMatrix1D(int n, Eigen::SparseMatrix<double>& H) {
    H.resize(n, n);
    H.makeCompressed();
    std::vector<Eigen::Triplet<double>> triplets;
    double scale = std::sqrt(1./ n);
    for (int i = 0; i < n; i++) {
        triplets.push_back({0, i, scale});
    }
    int row = 1;
    for (int nnz = n; nnz > 1; nnz /= 2) {
        int nrows = n / nnz;
        scale = std::sqrt(1./ nnz);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < nnz / 2; j++) {
                triplets.push_back({row, i*nnz + j, scale});
            }
            for (int j = nnz / 2; j < nnz; j++) {
                triplets.push_back({row, i*nnz + j, -scale});
            }
            row++;
        }
    }
    H.setFromTriplets(triplets.begin(), triplets.end());
}

void RectangleSurfaceSolver::FormMatrixCompressed(double threshold, bool print) {
    auto haarMesh = _unitMesh;
    haarMesh.HaarTransform();
    std::cout << "Forming truncated matrix\n";
    Profiler profiler;

    const auto& rectangles = haarMesh.Data();
    const int N = rectangles.size();
    std::vector<Eigen::Triplet<complex>> triplets;

    _truncMatrix.resize(_dim, _dim);
    _truncMatrix.makeCompressed();
    std::vector<size_t> rowStarts;
    rowStarts.push_back(0);

    size_t nnz = 0;
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            if (PlaneParRectDist(rectangles[j], rectangles[i]) < threshold) {
                triplets.push_back({2*i, 2*j, complex(0.)});
                triplets.push_back({2*i+1, 2*j, complex(0.)});
                triplets.push_back({2*i, 2*j+1, complex(0.)});
                triplets.push_back({2*i+1, 2*j+1, complex(0.)});
                nnz += 4;
            }
        }
        rowStarts.push_back(nnz);
    }

    Eigen::SparseMatrix<double> haarX, haarY;
    MakeHaarMatrix1D(_nx, haarX);
    MakeHaarMatrix1D(_ny, haarY);
    
    for (int k = 0; k < N; k++) {
        const int ky = k / _nx, kx = k % _nx;
        Eigen::MatrixXcd blockB;
        _formBlockCol(blockB, k);
        #pragma omp parallel for
        for (int jy = haarY.outerIndexPtr()[ky]; jy < haarY.outerIndexPtr()[ky+1]; jy++) {
            for (int jx = haarX.outerIndexPtr()[kx]; jx < haarX.outerIndexPtr()[kx+1]; jx++) {
                const int j = _nx * haarY.innerIndexPtr()[jy] + haarX.innerIndexPtr()[jx]; 
                const double haar = haarX.valuePtr()[jx] * haarY.valuePtr()[jy];
                for (size_t tr = rowStarts[j]; tr < rowStarts[j+1]; tr += 4) {
                    const int i = triplets[tr].row() / 2;

                    complex& C_0_0 = const_cast<complex&>(triplets[tr].value());
                    complex& C_1_0 = const_cast<complex&>(triplets[tr+1].value());
                    complex& C_0_1 = const_cast<complex&>(triplets[tr+2].value());
                    complex& C_1_1 = const_cast<complex&>(triplets[tr+3].value());

                    const auto B = blockB.block<2, 2>(2*i, 0);

                    C_0_0 += haar * B(0, 0);
                    C_1_0 += haar * B(1, 0);
                    C_0_1 += haar * B(0, 1);
                    C_1_1 += haar * B(1, 1);
                }
            }
        }
    }

    _truncMatrix.setFromTriplets(triplets.begin(), triplets.end());
    std::cout << "Time for forming truncated matrix: " << profiler.Toc() << " s.\n"; 
    std::cout << "Proportion of nonzeros: " << 1. * triplets.size() / _dim / _dim << "\n";

    if (print) {
        std::ofstream fout("trunc_mat.txt", std::ios::out);
        std::cout << "Printing truncated matrix" << '\n';
        for (const auto& triplet: triplets) {
            fout << triplet.col() << ' ' << triplet.row()
                 << ' ' << std::abs(triplet.value()) << '\n';
        }
        fout.close();    
    }
    std::cout << '\n';
}

void RectangleSurfaceSolver::FormRhs(const std::function<Eigen::Vector3cd(const Eigen::Vector3d&)>& f) {
    _rhs.resize(_dim);
    const auto& rectangles = _mesh.Data();
    for (int i = 0; i < rectangles.size(); i++) {
        const Eigen::Vector3cd b = f(rectangles[i].center);
        const auto& e1 = rectangles[i].e1.cast<complex>();
        const auto& e2 = rectangles[i].e2.cast<complex>();
        const auto& n =  rectangles[i].normal.cast<complex>();
        _rhs(2 * i    ) = -n.cross(b).dot(e2);
        _rhs(2 * i + 1) =  n.cross(b).dot(e1);
    }
}

void RectangleSurfaceSolver::PlotSolutionMap(Eigen::VectorXcd x) const {
    std::cout << "Applying inverse Haar transfrom\n";
    Subvector2D E0(x, _dim / 2, 0);
    HaarInverse2D(E0, _ny, _nx);
    Subvector2D E1(x, _dim / 2, 1);
    HaarInverse2D(E1, _ny, _nx);
    std::cout << "Printing solution\n";
    std::ofstream fout("solution.txt", std::ios::out);
    for (int i = 0; i < _dim; i++) {
        fout << std::abs(x(i)) << '\n';
    }
    fout.close();
}

void RectangleSurfaceSolver::HaarTransform() {
    if (_fullMatrix.size()) {
        for (int i = 0; i < _dim; i++) {
            auto col = _fullMatrix.col(i);
            Subvector2D E0(col, _dim / 2, 0);
            Haar2D(E0, _ny, _nx);
            Subvector2D E1(col, _dim / 2, 1);
            Haar2D(E1, _ny, _nx);
        }
        for (int i = 0; i < _dim; i++) {
            auto row = _fullMatrix.row(i);
            Subvector2D E0(row, _dim / 2, 0);
            Haar2D(E0, _ny, _nx);
            Subvector2D E1(row, _dim / 2, 1);
            Haar2D(E1, _ny, _nx);
        }
    }
    if (_rhs.size()) {
        Subvector2D f0(_rhs, _dim / 2, 0);
        Haar2D(f0, _ny, _nx);
        Subvector2D f1(_rhs, _dim / 2, 1);
        Haar2D(f1, _ny, _nx);
    }   
}

void RectangleSurfaceSolver::PrintFullMatrix(const std::string& file) const {
    std::ofstream fout(file, std::ios::out);
    for (int i = 0; i < _dim; i++) {
        for (int j = 0; j < _dim; j++) {
            fout << std::abs(_fullMatrix(i, j)) << ' ';
        }
        fout << '\n';
    }
}

void RectangleSurfaceSolver::PrintSolutionVtk(Eigen::VectorXcd x) const {
    std::cout << "Applying inverse Haar transfrom\n";
    Subvector2D E0(x, _dim / 2, 0);
    HaarInverse2D(E0, _ny, _nx);
    Subvector2D E1(x, _dim / 2, 1);
    HaarInverse2D(E1, _ny, _nx);
    _printVtk(x);
}

void RectangleSurfaceSolver::PrintEsa(const Eigen::VectorXcd& x) const {
    const int N = 360;
    std::ofstream fout("esa.txt", std::ios::out);
    for (int i = 0; i < N; ++i) {
        fout << _CalcEsa(x, 2 * M_PI * i / N) << '\n';
    }
    fout.close();
}

void RectangleSurfaceSolver::_printVtk(const Eigen::VectorXcd& x) const {
    std::ofstream fout("solution.vtk", std::ios::out);
    fout << "# vtk DataFile Version 3.0\n";
    fout << "Surface electric current\n";
    fout << "ASCII\n";
    fout << "DATASET POLYDATA\n";
    const int npoints = _mesh.Data().size() * 4;
    const int ncells  = _mesh.Data().size();
    fout << "POINTS " << npoints << " double\n";
    for (const auto& rectangle: _mesh.Data()) {
        fout << rectangle.a << '\n' << rectangle.b << '\n' << rectangle.c << '\n' << rectangle.d << '\n';
    }
    int i = 0;
    fout << "POLYGONS " << ncells << ' ' << 5 * ncells << '\n';
    for (const auto& rectangle: _mesh.Data()) {
        fout << "4 " << i << ' ' << i+1 << ' ' << i+2 << ' ' << i+3 << '\n';
        i += 4; 
    }
    fout << "CELL_DATA " << ncells << '\n';
    fout << "VECTORS J_REAL double\n";
    i = 0;
    for (const auto& rectangle: _mesh.Data()) {
        Eigen::Vector3cd J = x[2*i]*rectangle.e1.cast<complex>() + x[2*i+1]*rectangle.e2.cast<complex>();
        J = rectangle.normal.cast<complex>().cross(J).cross(rectangle.normal.cast<complex>());
        fout << J.real() << '\n';
        i++;
    }
    fout << "VECTORS J_IMAG double\n";
    i = 0;
    for (const auto& rectangle: _mesh.Data()) {
        Eigen::Vector3cd J = x[2*i]*rectangle.e1.cast<complex>() + x[2*i+1]*rectangle.e2.cast<complex>();
        J = rectangle.normal.cast<complex>().cross(J).cross(rectangle.normal.cast<complex>());
        fout << J.imag() << '\n';
        i++;
    }
    fout.close();
}

double RectangleSurfaceSolver::_CalcEsa(const Eigen::VectorXcd& x, double phi) const {
    Eigen::Vector3d tau;
    tau << 0, std::cos(phi), std::sin(phi);
    const double c = 3e8;
    const double eps = 8.85e-12;
    int i = 0;
    Eigen::Vector3cd sigma;
    sigma << 0., 0., 0.;
    for (const auto& rectangle: _mesh.Data()) {
        Eigen::Vector3cd J = x[2*i]*rectangle.e1.cast<complex>() + x[2*i+1]*rectangle.e2.cast<complex>();
        J = rectangle.normal.cast<complex>().cross(J).cross(rectangle.normal.cast<complex>());
        const auto& y = rectangle.center;
        const double ds = rectangle.area;
        sigma += std::exp(-1i*_k*tau.dot(y)) * 1i * _k / c / eps * ds * 
            (J - J.dot(tau.cast<complex>()) * tau.cast<complex>());
        i++;
    }
    return 10. * std::log10(4 * M_PI * sigma.norm() * sigma.norm());
}


inline double PlaneParRectDist(const Rectangle& A, const Rectangle& B) {
    const double dx = Distance({A.a[0], A.b[0]}, {B.a[0], B.b[0]});
    const double dy = Distance({A.b[1], A.c[1]}, {B.b[1], B.c[1]});
    return std::sqrt(dx*dx + dy*dy);
}

SurfaceSolver::SurfaceSolver(
    double k, 
    const std::string& meshFile, 
    const std::string& graphFile
): RectangleSurfaceSolver(k) {

    _mesh = RectangleMesh(meshFile, graphFile);
    _dim = 2 * _mesh.Data().size(); 
}

void SurfaceSolver::WaveletTransform() {
    _mesh.FormWaveletMatrix();
}

void SurfaceSolver::PrintEsa(const Eigen::VectorXcd& x) const {
    const int N = 360;
    std::ofstream fout("esa.txt", std::ios::out);
    for (int i = 0; i < N; ++i) {
        fout << _CalcEsa(x, 2 * M_PI * i / N) << '\n';
    }
    fout.close();
}

double SurfaceSolver::_CalcEsa(const Eigen::VectorXcd& x, double phi) const {
    Eigen::Vector3d tau;
    tau << std::cos(phi), std::sin(phi), 0.;
    const double c = 3e8;
    const double eps = 8.85e-12;
    int i = 0;
    Eigen::Vector3cd sigma;
    sigma << 0., 0., 0.;
    for (const auto& rectangle: _mesh.Data()) {
        Eigen::Vector3cd J = x[2*i]*rectangle.e1.cast<complex>() + x[2*i+1]*rectangle.e2.cast<complex>();
        J = rectangle.normal.cast<complex>().cross(J).cross(rectangle.normal.cast<complex>());
        const auto& y = rectangle.center;
        const double ds = rectangle.area;
        sigma += std::exp(-1i*_k*tau.dot(y)) * 1i * _k / c / eps * ds * 
            (J - J.dot(tau.cast<complex>()) * tau.cast<complex>());
        i++;
    }
    return 10. * std::log10(4 * M_PI * sigma.norm() * sigma.norm());
}

}