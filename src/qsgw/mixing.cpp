#include "mixing.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

// Legacy code path: call LAPACK dgetrf/dgetrs explicitly (historical QSGW traces).
// Self-declared prototypes keep this TU independent of lapack_connector.h, matching
// the original pulay_mixing.cpp. LAPACK is linked for the whole rpa_lib target.
extern "C" {
    void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
    void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a, const int* lda,
                 const int* ipiv, double* b, const int* ldb, int* info);
    double dlange_(const char* norm, const int* m, const int* n, const double* a, const int* lda,
                   double* work);
    void dgecon_(const char* norm, const int* n, const double* a, const int* lda, const double* anorm,
                 double* rcond, double* work, int* iwork, int* info);
}

namespace librpa_int {

namespace {
bool pulay_debug_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("LIBRPA_PULAY_DEBUG");
        cached = (env != nullptr && std::string(env) != "0") ? 1 : 0;
    }
    return cached == 1;
}

double matrix_l2_norm(const matrix& M) {
    double sum = 0.0;
    for (int i = 0; i < M.size; i++) {
        sum += M.c[i] * M.c[i];
    }
    return std::sqrt(sum);
}

double matrix_max_abs(const matrix& M) {
    double vmax = 0.0;
    for (int i = 0; i < M.size; i++) {
        vmax = std::max(vmax, std::abs(M.c[i]));
    }
    return vmax;
}

std::string fmt_sci(double v) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(15) << v;
    return os.str();
}

double pulay_residual_quantize_scale() {
    static bool initialized = false;
    static double scale = 0.0;
    if (!initialized) {
        initialized = true;
        const char* env = std::getenv("LIBRPA_PULAY_RESIDUAL_QUANTIZE_SCALE");
        if (env != nullptr) {
            try {
                scale = std::stod(std::string(env));
            } catch (...) {
                scale = 0.0;
            }
        }
        if (!(scale > 0.0)) {
            scale = 0.0;
        }
    }
    return scale;
}

double pulay_b_quantize_scale() {
    static bool initialized = false;
    static double scale = 0.0;
    if (!initialized) {
        initialized = true;
        const char* env = std::getenv("LIBRPA_PULAY_B_QUANTIZE_SCALE");
        if (env != nullptr) {
            try {
                scale = std::stod(std::string(env));
            } catch (...) {
                scale = 0.0;
            }
        }
        if (!(scale > 0.0)) {
            scale = 0.0;
        }
    }
    return scale;
}

int pulay_residual_quantize_start_step() {
    static bool initialized = false;
    static int start_step = 1;
    if (!initialized) {
        initialized = true;
        const char* env = std::getenv("LIBRPA_PULAY_RESIDUAL_QUANTIZE_START_STEP");
        if (env != nullptr) {
            try {
                start_step = std::stoi(std::string(env));
            } catch (...) {
                start_step = 1;
            }
        }
        if (start_step < 1) {
            start_step = 1;
        }
    }
    return start_step;
}

int pulay_b_quantize_start_step() {
    static bool initialized = false;
    static int start_step = 1;
    if (!initialized) {
        initialized = true;
        const char* env = std::getenv("LIBRPA_PULAY_B_QUANTIZE_START_STEP");
        if (env != nullptr) {
            try {
                start_step = std::stoi(std::string(env));
            } catch (...) {
                start_step = 1;
            }
        }
        if (start_step < 1) {
            start_step = 1;
        }
    }
    return start_step;
}

double pulay_alpha_quantize_scale() {
    static bool initialized = false;
    static double scale = 0.0;
    if (!initialized) {
        initialized = true;
        const char* env = std::getenv("LIBRPA_PULAY_ALPHA_QUANTIZE_SCALE");
        if (env != nullptr) {
            try {
                scale = std::stod(std::string(env));
            } catch (...) {
                scale = 0.0;
            }
        }
        if (!(scale > 0.0)) {
            scale = 0.0;
        }
    }
    return scale;
}

int pulay_alpha_quantize_start_step() {
    static bool initialized = false;
    static int start_step = 1;
    if (!initialized) {
        initialized = true;
        const char* env = std::getenv("LIBRPA_PULAY_ALPHA_QUANTIZE_START_STEP");
        if (env != nullptr) {
            try {
                start_step = std::stoi(std::string(env));
            } catch (...) {
                start_step = 1;
            }
        }
        if (start_step < 1) {
            start_step = 1;
        }
    }
    return start_step;
}

void quantize_matrix_inplace(matrix& M, double scale) {
    if (!(scale > 0.0)) {
        return;
    }
    for (int i = 0; i < M.size; i++) {
        M.c[i] = std::round(M.c[i] * scale) / scale;
    }
}
}  // namespace

// ============================================================================
// Legacy Pulay/DIIS mixer (fixed beta, no adaptive/limiters)
// Enabled for reproducibility of historical QSGW iteration traces.
//
// The newer enhanced mixer is preserved in the legacy tree as
//   qsgw/pulay_mixing_enhanced.cpp
// (disabled via #if 0 in the original) and can be re-enabled by swapping
// implementations. The active 7a7ff17f path is this fixed-beta version.
// ============================================================================

PulayMixer::PulayMixer(int max_history, double mixing_beta)
    : max_history_(max_history), current_step_(0), mixing_beta_(mixing_beta),
      initialized_(false), input_history_(), residual_history_(), nrows_(0), ncols_(0),
      adaptive_enabled_(false), beta_min_(0.0), beta_max_(0.0), residual_history_norms_(),
      eigenvalue_change_history_(), last_beta_adjustment_step_(0) {}

void PulayMixer::initialize(const matrix& initial_guess) {
    nrows_ = initial_guess.nr;
    ncols_ = initial_guess.nc;
    input_history_.clear();
    residual_history_.clear();

    input_history_.push_back(initial_guess);
    initialized_ = true;
    current_step_ = 0;

    std::cout << "[PulayMixer] Initialized with matrix of size " << nrows_ << "x" << ncols_
              << std::endl;
}

matrix PulayMixer::mix(const matrix& current_output) {
    if (!initialized_) {
        throw std::runtime_error("[PulayMixer] Not initialized. Call initialize() first.");
    }
    if (current_output.nr != nrows_ || current_output.nc != ncols_) {
        throw std::runtime_error("[PulayMixer] Matrix dimensions do not match initialization.");
    }

    current_step_++;

    matrix current_residual = current_output - input_history_.back();
    const double residual_quantize_scale = pulay_residual_quantize_scale();
    const int residual_quantize_start_step = pulay_residual_quantize_start_step();
    if (residual_quantize_scale > 0.0 && current_step_ >= residual_quantize_start_step) {
        quantize_matrix_inplace(current_residual, residual_quantize_scale);
    }
    residual_history_.push_back(current_residual);

    if ((int)residual_history_.size() > max_history_) {
        residual_history_.erase(residual_history_.begin());
        input_history_.erase(input_history_.begin());
    }

    int history_size = (int)residual_history_.size();
    bool use_linear_mixing = (history_size <= 1);
    matrix alpha;

    if (pulay_debug_enabled()) {
        std::cout << "[PulayDebug] step=" << current_step_ << " history_size=" << history_size
                  << " residual_l2=" << fmt_sci(matrix_l2_norm(current_residual))
                  << " residual_max=" << fmt_sci(matrix_max_abs(current_residual))
                  << " residual_quant_scale=" << fmt_sci(residual_quantize_scale)
                  << " residual_quant_start_step=" << residual_quantize_start_step << std::endl;
    }

    if (!use_linear_mixing) {
        try {
            matrix B(history_size + 1, history_size + 1, true);
            matrix rhs(history_size + 1, 1, true);

            for (int i = 0; i < history_size; i++) {
                for (int j = i; j < history_size; j++) {
                    double inner_prod =
                        matrix_inner_product(residual_history_[i], residual_history_[j]);
                    B(i, j) = inner_prod;
                    B(j, i) = inner_prod;
                }
                B(i, history_size) = -1.0;
                B(history_size, i) = -1.0;
            }
            B(history_size, history_size) = 0.0;
            rhs(history_size, 0) = -1.0;

            const double b_quantize_scale = pulay_b_quantize_scale();
            const int b_quantize_start_step = pulay_b_quantize_start_step();
            if (b_quantize_scale > 0.0 && current_step_ >= b_quantize_start_step) {
                quantize_matrix_inplace(B, b_quantize_scale);
                for (int i = 0; i < history_size; i++) {
                    B(i, history_size) = -1.0;
                    B(history_size, i) = -1.0;
                }
                B(history_size, history_size) = 0.0;
            }

            if (pulay_debug_enabled()) {
                double bdiag_min = std::numeric_limits<double>::infinity();
                double bdiag_max = 0.0;
                for (int i = 0; i < history_size; i++) {
                    double ad = std::abs(B(i, i));
                    bdiag_min = std::min(bdiag_min, ad);
                    bdiag_max = std::max(bdiag_max, ad);
                }
                std::cout << "[PulayDebug] step=" << current_step_
                          << " B_diag_abs_min=" << fmt_sci(bdiag_min)
                          << " B_diag_abs_max=" << fmt_sci(bdiag_max)
                          << " B_quant_scale=" << fmt_sci(b_quantize_scale)
                          << " B_quant_start_step=" << b_quantize_start_step << std::endl;
            }

            alpha = solve_linear_system(B, rhs);

            const double alpha_quantize_scale = pulay_alpha_quantize_scale();
            const int alpha_quantize_start_step = pulay_alpha_quantize_start_step();
            if (alpha_quantize_scale > 0.0 && current_step_ >= alpha_quantize_start_step) {
                quantize_matrix_inplace(alpha, alpha_quantize_scale);
            }

            if (pulay_debug_enabled()) {
                double alpha_sum = 0.0;
                std::ostringstream avec;
                avec << "[";
                for (int i = 0; i < history_size; i++) {
                    double ai = alpha(i, 0);
                    alpha_sum += ai;
                    if (i) {
                        avec << ",";
                    }
                    avec << fmt_sci(ai);
                }
                avec << "]";
                std::cout << "[PulayDebug] step=" << current_step_ << " alpha_sum=" << fmt_sci(alpha_sum)
                          << " alpha_quant_scale=" << fmt_sci(alpha_quantize_scale)
                          << " alpha_quant_start_step=" << alpha_quantize_start_step
                          << " alpha=" << avec.str() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[PulayMixer] Warning: Pulay mixing failed (" << e.what()
                      << "). Resetting history and falling back to linear mixing." << std::endl;

            while ((int)residual_history_.size() > 1) {
                residual_history_.erase(residual_history_.begin());
                input_history_.erase(input_history_.begin());
            }
            history_size = 1;
            use_linear_mixing = true;
        }
    }

    if (use_linear_mixing) {
        matrix new_input = input_history_.back() + mixing_beta_ * current_residual;
        if (pulay_debug_enabled()) {
            std::cout << "[PulayDebug] step=" << current_step_ << " mode=linear beta=" << fmt_sci(mixing_beta_)
                      << " delta_input_l2=" << fmt_sci(matrix_l2_norm(new_input - input_history_.back()))
                      << std::endl;
        }
        input_history_.push_back(new_input);
        std::cout << "[PulayMixer] Performed simple linear mixing." << std::endl;
        return new_input;
    }

    matrix new_input(nrows_, ncols_, true);
    for (int i = 0; i < history_size; i++) {
        matrix term = input_history_[i] + mixing_beta_ * residual_history_[i];
        new_input += alpha(i, 0) * term;
    }

    if (pulay_debug_enabled()) {
        matrix delta = new_input - input_history_.back();
        std::cout << "[PulayDebug] step=" << current_step_ << " mode=pulay beta=" << fmt_sci(mixing_beta_)
                  << " delta_input_l2=" << fmt_sci(matrix_l2_norm(delta))
                  << " delta_input_max=" << fmt_sci(matrix_max_abs(delta)) << std::endl;
    }

    input_history_.push_back(new_input);
    std::cout << "[PulayMixer] Performed Pulay mixing at step " << current_step_ << "."
              << std::endl;
    return new_input;
}

matrix PulayMixer::mix(const matrix& current_output, double /*eigenvalue_change_ev*/) {
    return mix(current_output);
}

int PulayMixer::get_history_size() const {
    return (int)residual_history_.size();
}

int PulayMixer::get_current_step() const {
    return current_step_;
}

void PulayMixer::reset() {
    initialized_ = false;
    input_history_.clear();
    residual_history_.clear();
    nrows_ = 0;
    ncols_ = 0;
    current_step_ = 0;

    std::cout << "[PulayMixer] Reset mixer." << std::endl;
}

void PulayMixer::set_mixing_beta(double beta) {
    mixing_beta_ = beta;
}

double PulayMixer::get_mixing_beta() const {
    return mixing_beta_;
}

PulayMixerState PulayMixer::snapshot() const {
    PulayMixerState state;
    state.max_history = max_history_;
    state.current_step = current_step_;
    state.mixing_beta = mixing_beta_;
    state.initialized = initialized_;
    state.nrows = nrows_;
    state.ncols = ncols_;
    state.input_history = input_history_;
    state.residual_history = residual_history_;
    return state;
}

void PulayMixer::restore(const PulayMixerState& state) {
    max_history_ = state.max_history;
    current_step_ = state.current_step;
    mixing_beta_ = state.mixing_beta;
    initialized_ = state.initialized;
    nrows_ = state.nrows;
    ncols_ = state.ncols;
    input_history_ = state.input_history;
    residual_history_ = state.residual_history;
    residual_history_norms_.clear();
    eigenvalue_change_history_.clear();
    last_beta_adjustment_step_ = 0;
}

double PulayMixer::matrix_inner_product(const matrix& A, const matrix& B) {
    if (A.nr != B.nr || A.nc != B.nc) {
        throw std::runtime_error("[PulayMixer] Matrix dimensions must match for inner product.");
    }

    double result = 0.0;
    for (int i = 0; i < A.size; i++) {
        result += A.c[i] * B.c[i];
    }
    return result;
}

matrix PulayMixer::solve_linear_system(const matrix& A, const matrix& b) {
    if (A.nr != A.nc) {
        throw std::runtime_error("[PulayMixer] Matrix A must be square for linear system solving.");
    }
    if (b.nc != 1) {
        throw std::runtime_error("[PulayMixer] Right-hand side must be a column vector.");
    }
    if (A.nr != b.nr) {
        throw std::runtime_error("[PulayMixer] Matrix A and vector b must have compatible dimensions.");
    }

    const int n = A.nr;
    matrix A_copy = A;
    matrix x = b;

    int* ipiv = new int[n];
    int info = 0;
    const int nrhs = 1;

    // Prefer the explicit LU+solve path (dgetrf + dgetrs) used in the historical mixer.
    dgetrf_(&n, &n, A_copy.c, &n, ipiv, &info);
    if (info != 0) {
        delete[] ipiv;
        throw std::runtime_error("[PulayMixer] LU factorization failed.");
    }

    if (pulay_debug_enabled()) {
        double udiag_min = std::numeric_limits<double>::infinity();
        double udiag_max = 0.0;
        for (int i = 0; i < n; i++) {
            double ad = std::abs(A_copy(i, i));
            udiag_min = std::min(udiag_min, ad);
            udiag_max = std::max(udiag_max, ad);
        }

        const char norm = '1';
        std::vector<double> work_lange(std::max(1, n), 0.0);
        const double anorm = dlange_(&norm, &n, &n, A.c, &n, work_lange.data());

        double rcond = -1.0;
        int info_con = 0;
        std::vector<double> work_con(std::max(1, 4 * n), 0.0);
        std::vector<int> iwork_con(std::max(1, n), 0);
        dgecon_(&norm, &n, A_copy.c, &n, &anorm, &rcond, work_con.data(), iwork_con.data(), &info_con);

        const double cond_est =
            (info_con == 0 && rcond > 0.0) ? (1.0 / rcond) : std::numeric_limits<double>::infinity();

        std::cout << "[PulayDebug] LU diag_abs_min=" << fmt_sci(udiag_min)
                  << " diag_abs_max=" << fmt_sci(udiag_max)
                  << " anorm1=" << fmt_sci(anorm)
                  << " rcond1=" << fmt_sci(rcond)
                  << " cond1_est=" << fmt_sci(cond_est)
                  << " dgecon_info=" << info_con << std::endl;
    }

    const char trans = 'N';
    dgetrs_(&trans, &n, &nrhs, A_copy.c, &n, ipiv, x.c, &n, &info);

    delete[] ipiv;

    if (info != 0) {
        throw std::runtime_error("[PulayMixer] Linear system solving failed.");
    }

    return x;
}

} // namespace librpa_int
