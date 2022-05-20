#include "timefreq.h"
#include "mathtools.h"
#include "envs.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <algorithm>

using std::map;
using std::pair;
using std::string;
using std::vector;
using std::ifstream;
using std::endl;
using std::cout;

const string minimax_grid_path = string(source_dir) + "/minimax_grid";
const string GX_path = minimax_grid_path + "/GreenX/generate_local_grid.py";

map<double, double> read_local_grid(int grid_N, const string &file_path, const char type, double scale)
{
    ifstream infile;
    map<double, double> grid;
    infile.open(file_path);

    vector<double> tran(grid_N * 2);
    string ss;
    int itran = 0;
    while (infile.peek() != EOF)
    {
        if (infile.peek() == EOF)
            break;
        infile >> ss;
        tran[itran] = stod(ss);
        itran++;
    }
    for (int i = 0; i != grid_N; i++)
    {
        grid.insert(pair<double, double>(tran[i], tran[i + grid_N]));
    }

    infile.close();
    map<double, double> minimax_grid;
    minimax_grid.clear();
    switch (type)
    {
    case 'F':
    {
        if (grid_N <= 20)
        {
            for (auto &i_pair : grid)
                minimax_grid.insert({i_pair.first * scale, i_pair.second * scale * 0.25});
        }
        else
        {
            for (auto &i_pair : grid)
                minimax_grid.insert({i_pair.first * scale, i_pair.second * scale});
        }

        cout << " MINIMAX_GRID_Freq " << endl;
        for (const auto &m : minimax_grid)
            cout << m.first << "      " << m.second << endl;
        break;
    }

    case 'T':
    {
        for (auto i_pair : grid)
            minimax_grid.insert({i_pair.first / (scale), i_pair.second / (scale)});

        cout << " MINIMAX_GRID_Tau " << endl;
        for (const auto &m : minimax_grid)
            cout << m.first << "      " << m.second << endl;
        break;
    }
    }

    return minimax_grid;
}

vector<double> read_time2freq_trans(const string &file_path, double inverse_scale)
{
    ifstream infile;
    infile.open(file_path);

    vector<double> tran;
    string s;
    int n_freq = 0;
    // stringstream ss;
    // double ss_d;
    while (getline(infile, s))
    {
        if (s[0] == 'F')
        {
            n_freq++;
        }
        else
        {
            stringstream ss(s);
            double ss_d;
            ss >> ss_d;
            tran.push_back(ss_d);
        }
    }
    infile.close();

    double gap;
    double Emin, Emax;
    /* cout << "Cosine_tran_grid" << endl; */
    cout << "read transformation grid" << endl;
    for (int i = 0; i != tran.size(); i++)
    {
        tran[i] /= inverse_scale;
        // cout<<tran[i]<<endl;
    }
    return tran;
}

const string TFGrids::GRID_TYPES_NOTES[TFGrids::GRID_TYPES::COUNT] =
    {
        "Gauss-Legendre grids",
        "Gauss-Chebyshev grids of the first kind",
        "Gauss-Chebyshev grids of the second kind",
        "Minimax time-frequency grids",
        "Even-spaced frequency grids",
        "Even-spaced time-frequency grids (debug use)",
    };

const bool TFGrids::SUPPORT_TIME_GRIDS[TFGrids::GRID_TYPES::COUNT] = 
    { false, false, false, true, false, true };

void TFGrids::set_freq()
{
    freq_nodes.resize(n_grids);
    freq_weights.resize(n_grids);
}

void TFGrids::set_time()
{
    time_nodes.resize(n_grids);
    time_weights.resize(n_grids);
    costrans_t2f.create(n_grids, n_grids);
    sintrans_t2f.create(n_grids, n_grids);
    fourier_t2f.create(n_grids, n_grids);
}

void TFGrids::show()
{
    cout << "Grid type: " << TFGrids::GRID_TYPES_NOTES[grid_type] << endl;
    cout << "Grid size: " << n_grids << endl;
    cout << "Frequency node & weight: " << endl;
    for ( int i = 0; i != n_grids; i++ )
        printf("%2d %10.6f %10.6f\n", i, freq_nodes[i], freq_weights[i]);
    if (has_time_grids())
    {
        cout << "Time node & weight: " << endl;
        for ( int i = 0; i != n_grids; i++ )
            printf("%2d %10.6f %10.6f\n", i, time_nodes[i], time_weights[i]);
        cout << "t->f transform: " << endl;
        if (costrans_t2f.size)
        {
            cout << "Cosine transform matrix" << endl;
            print_matrix("", costrans_t2f);
        }
        if (sintrans_t2f.size)
        {
            cout << "Sine transform matrix" << endl;
            print_matrix("", sintrans_t2f);
        }
    }
}

void TFGrids::unset()
{
    freq_nodes.clear();
    freq_weights.clear();
    time_nodes.clear();
    time_weights.clear();
    costrans_t2f.create(0, 0);
    sintrans_t2f.create(0, 0);
    fourier_t2f.create(0, 0);
}

TFGrids::TFGrids(unsigned N)
{
    n_grids = N;
    set_freq();
}

void TFGrids::reset(unsigned N)
{
    unset();
    n_grids = N;
    set_freq();
}

TFGrids::~TFGrids()
{
    /* unset(); */
}

void TFGrids::generate_evenspaced(double emin, double interval)
{
    if ( emin <= 0 )
        throw invalid_argument("emin must be positive");
    if ( interval < 0 )
        throw invalid_argument("emin must be non-negative");
    double weight = 1.0 / n_grids;
    for ( int i = 0; i != n_grids; i++)
    {
        freq_nodes[i] = emin + interval * i;
        freq_weights[i] = weight;
    }
}

void TFGrids::generate_evenspaced_tf(double emin, double eintv, double tmin, double tintv)
{
    generate_evenspaced(emin, eintv);
    set_time();
    if ( tmin <= 0 )
        throw invalid_argument("tmin must be positive");
    if ( tintv < 0 )
        throw invalid_argument("tintv must be non-negative");
    double weight = 1.0 / n_grids;
    for ( int i = 0; i != n_grids; i++)
    {
        time_nodes[i] = tmin + tintv * i;
        time_weights[i] = weight;
        // WARN: fake transform matrices
        costrans_t2f(i, i) = weight;
        sintrans_t2f(i, i) = weight;
    }
}

void TFGrids::generate_minimax(double emin, double emax)
{
    grid_type = TFGrids::GRID_TYPES::Minimax;
    set_time();
    string tmps;
    if ( emin <= 0)
        throw invalid_argument("emin must be positive");
    if ( emax < emin)
        throw invalid_argument("emax must be larger than emin");
    double erange = emax / emin;
    tmps = "python " + GX_path + " " + to_string(n_grids) + " " + to_string(erange);
    system(tmps.c_str());

    map<double, double> freq_grid = read_local_grid(n_grids, "local_" + to_string(n_grids) + "_freq_points.dat", 'F', emin);
    int ig = 0;
    for (auto nw: freq_grid)
    {
        freq_nodes[ig] = nw.first;
        freq_weights[ig] = nw.second;
        ig++;
    }
    map<double, double> time_grid = read_local_grid(n_grids, "local_" + to_string(n_grids) + "_time_points.dat", 'T', emin);
    ig = 0;
    for (auto nw: time_grid)
    {
        time_nodes[ig] = nw.first;
        time_weights[ig] = nw.second;
        ig++;
    }
    vector<double> trans;
    // cosine transform
    trans = read_time2freq_trans(to_string(n_grids) + "_time2freq_grid_cos.txt", emin);
    for (int k = 0; k != n_grids; k++)
        for (int j = 0; j != n_grids; j++)
            costrans_t2f(k, j) = trans[ k * n_grids + j];
    trans = read_time2freq_trans(to_string(n_grids) + "_time2freq_grid_sin.txt", emin);
    for (int k = 0; k != n_grids; k++)
        for (int j = 0; j != n_grids; j++)
            sintrans_t2f(k, j) = trans[ k * n_grids + j];
}

void TFGrids::generate_GaussChebyshevI()
{
    grid_type = TFGrids::GRID_TYPES::GaussChebyshevI;
    double nodes[n_grids], weights[n_grids];
    GaussChebyshevI_unit(n_grids, nodes, weights);
    // transform from [-1,1] to [0, infinity]
    transform_GaussQuad_unit2x0inf(0.0, n_grids, nodes, weights);
    for ( int i = 0; i != n_grids; i++ )
    {
        freq_nodes[i] = nodes[i];
        freq_weights[i] = weights[i];
    }
}

void TFGrids::generate_GaussChebyshevII()
{
    grid_type = TFGrids::GRID_TYPES::GaussChebyshevII;
    double nodes[n_grids], weights[n_grids];
    GaussChebyshevII_unit(n_grids, nodes, weights);
    // transform from [-1,1] to [0, infinity]
    transform_GaussQuad_unit2x0inf(0.0, n_grids, nodes, weights);
    for ( int i = 0; i != n_grids; i++ )
    {
        freq_nodes[i] = nodes[i];
        freq_weights[i] = weights[i];
    }
}

void TFGrids::generate_GaussLegendre()
{
    grid_type = TFGrids::GRID_TYPES::GaussLegendre;
    double nodes[n_grids], weights[n_grids];
    GaussLegendre_unit(n_grids, nodes, weights);
    // transform from [-1,1] to [0, infinity]
    transform_GaussQuad_unit2x0inf(0.0, n_grids, nodes, weights);
    for ( int i = 0; i != n_grids; i++ )
    {
        freq_nodes[i] = nodes[i];
        freq_weights[i] = weights[i];
    }
}

double TFGrids::find_freq_weight(const double & freq) const
{
    auto itr = std::find(freq_nodes.begin(), freq_nodes.end(), freq);
    if ( itr == freq_nodes.end() )
        throw invalid_argument("frequency not found");
    int i = std::distance(freq_nodes.begin(), itr);
    return freq_weights[i];
}
