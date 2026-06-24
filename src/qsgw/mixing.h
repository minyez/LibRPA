#ifndef PULAY_MIXER_H
#define PULAY_MIXER_H

#include "../math/matrix.h"
#include <vector>
#include <deque>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace librpa_int {

// NOTE(focus-window): The legacy driver also declared `mix_focus_nbands` /
// `mix_focus_outside_damp` (task_qsgw.cpp:961/963) for a "focus-window weighted
// Hamiltonian mixing", but a full code audit of 7a7ff17f shows those knobs were
// only declared + parsed and NEVER consumed at the mixing call site — the real
// H0_GW mix packs every spin/kpoint/band uniformly and calls mix() with no
// per-band weighting (LEADER_AUDIT §4). So this mixer faithfully matches the old
// active path by NOT implementing focus-window weighting. `linear_mixing_steps`
// is likewise a driver-level (and in 7a7ff17f a no-op) concept, not a class param.

struct PulayMixerState {
    int max_history = 0;
    int current_step = 0;
    double mixing_beta = 0.0;
    bool initialized = false;
    int nrows = 0;
    int ncols = 0;
    std::vector<matrix> input_history;
    std::vector<matrix> residual_history;
};

class PulayMixer {
private:
    int max_history_;           // maximum history length
    int current_step_;          // current step index
    double mixing_beta_;        // mixing parameter
    bool initialized_;          // initialized flag

    // history storage
    std::vector<matrix> input_history_;   // input matrix history
    std::vector<matrix> residual_history_; // residual matrix history

    // matrix dimensions
    int nrows_;
    int ncols_;

    // adaptive parameters
    bool adaptive_enabled_;               // enable adaptive tuning
    double beta_min_;                    // minimal beta
    double beta_max_;                    // maximal beta
    std::deque<double> residual_history_norms_;
    std::deque<double> eigenvalue_change_history_;  // residual-norm history (for adaptive analysis)
    int last_beta_adjustment_step_;      // last step beta was adjusted

    // private member function declarations
    double get_adaptive_beta();  // adaptive beta tuning
    double matrix_inner_product(const matrix& A, const matrix& B);
    matrix solve_linear_system(const matrix& A, const matrix& b);

public:
    // constructor
    // Defaults (5, 0.1) match the legacy pulay_mixing.h; the QSGW driver actually
    // constructs with (12, 0.2) (task_qsgw.cpp:970/969/1051).
    PulayMixer(int max_history = 5, double mixing_beta = 0.1);

    /**
     * @brief Initialize the mixer; must be called before the first mix()
     * @param initial_guess initial guess matrix
     */
    void initialize(const matrix& initial_guess);

    /**
     * @brief Perform Pulay mixing
     * @param current_output output matrix of the current iteration
     * @return the mixed new input matrix
     */
    matrix mix(const matrix& current_output);
    matrix mix(const matrix& current_output, double eigenvalue_change_ev);

    /**
     * @brief Return current history size
     */
    int get_history_size() const;

    /**
     * @brief Return current step index
     */
    int get_current_step() const;

    /**
     * @brief Reset the mixer
     */
    void reset();

    /**
     * @brief Set the mixing parameter
     */
    void set_mixing_beta(double beta);

    /**
     * @brief Return the mixing parameter
     */
    double get_mixing_beta() const;

    /**
     * @brief Enable/disable adaptive tuning
     */
    void set_adaptive_enabled(bool enabled) { adaptive_enabled_ = enabled; }

    /**
     * @brief Set the beta range
     */
    void set_beta_bounds(double min_beta, double max_beta) {
        beta_min_ = min_beta;
        beta_max_ = max_beta;
    }

    /**
     * @brief Return the residual-norm history
     */
    const std::deque<double>& get_residual_history() const {
        return residual_history_norms_;
    }

    /**
     * @brief Export/restore the minimal state needed to resume the mixer
     */
    PulayMixerState snapshot() const;
    void restore(const PulayMixerState& state);
};

} // namespace librpa_int

#endif // PULAY_MIXER_H
