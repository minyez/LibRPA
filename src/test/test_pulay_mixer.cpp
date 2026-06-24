// Unit test for the QSGW Pulay mixer (src/qsgw/mixing.{h,cpp}).
// Faithfully ported from legacy qsgw/pulay_mixing.{h,cpp}; the API/logic is
// unchanged, only placed in namespace librpa_int and built against the new
// matrix type. Exercises: linear mixing path, full Pulay path (LAPACK solve),
// and snapshot/restore for checkpointing.
#include "../qsgw/mixing.h"
#include "../math/matrix.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace librpa_int;

static bool mat_equal(const matrix &a, const matrix &b, double tol)
{
    if (a.nr != b.nr || a.nc != b.nc) return false;
    for (int i = 0; i < a.size; ++i)
        if (std::fabs(a.c[i] - b.c[i]) > tol) return false;
    return true;
}

static double mat_max_abs_diff(const matrix &a, const matrix &b)
{
    double m = 0.0;
    for (int i = 0; i < a.size; ++i)
        m = std::max(m, std::fabs(a.c[i] - b.c[i]));
    return m;
}

// Fixed-point map f(x) = x_star + alpha * (x - x_star), a contraction toward
// x_star with factor alpha in (0,1). A working Pulay mixer must drive the
// iterated input to x_star.
static matrix fixed_point_map(const matrix &x, const matrix &x_star, double alpha)
{
    // f(x) = x_star + alpha*(x - x_star) = (1-alpha)*x_star + alpha*x
    matrix out = x_star;            // (1-alpha) handled below via scaling
    out *= (1.0 - alpha);
    matrix ax = x;
    ax *= alpha;
    out += ax;
    return out;
}

// 1. First-step linear mixing:
//    residual = out - x0;  new = x0 + beta*residual = x0 + beta*(out - x0).
static void test_linear_first_step()
{
    const double beta = 0.2;
    PulayMixer mixer(12, beta);
    matrix x0(2, 2, true);
    x0(0, 0) = 1.0; x0(1, 1) = 2.0;
    mixer.initialize(x0);

    matrix out(2, 2, true);
    out(0, 0) = 5.0; out(1, 1) = -3.0;

    matrix mixed = mixer.mix(out);

    matrix expect = x0;
    // expect = x0 + beta*(out - x0)
    matrix diff = out - x0;
    diff *= beta;
    expect += diff;

    assert(mat_equal(mixed, expect, 1e-12));
    assert(mixer.get_current_step() == 1);
    assert(mixer.get_history_size() == 1);
    std::cout << "[OK] test_linear_first_step\n";
}

// 2. Full Pulay path drives a contraction to its fixed point.
//    Requires the LAPACK linear solve inside PulayMixer::solve_linear_system.
static void test_pulay_converges_to_fixed_point()
{
    const double alpha = 0.5;   // contraction factor
    const double beta = 0.2;
    matrix x_star(3, 3, true);
    x_star(0, 0) = 1.0;
    x_star(1, 1) = 2.0;
    x_star(2, 2) = 3.0;
    // small off-diagonal target so the matrix is non-trivial
    x_star(0, 1) = 0.25;
    x_star(1, 0) = 0.25;

    PulayMixer mixer(12, beta);
    matrix x(3, 3, true);        // initial guess = 0
    mixer.initialize(x);

    double last_err = 1e9;
    for (int step = 0; step < 30; ++step)
    {
        matrix out = fixed_point_map(x, x_star, alpha);
        x = mixer.mix(out);
        last_err = mat_max_abs_diff(x, x_star);
    }
    std::cout << "[info] pulay final max-abs-diff to fixed point: " << last_err << "\n";
    assert(last_err < 1e-6);
    std::cout << "[OK] test_pulay_converges_to_fixed_point\n";
}

// 3. snapshot/restore round-trip reproduces mixer state.
static void test_snapshot_restore()
{
    PulayMixer a(12, 0.2);
    matrix x0(2, 2, true);
    x0(0, 0) = 0.5; x0(1, 1) = -0.5;
    a.initialize(x0);
    matrix x_star(2, 2, true);
    x_star(0, 0) = 1.0; x_star(1, 1) = 1.0;
    // advance a couple of steps to populate history
    matrix x = x0;
    for (int i = 0; i < 4; ++i)
        x = a.mix(fixed_point_map(x, x_star, 0.4));

    PulayMixerState st = a.snapshot();

    PulayMixer b(5, 0.1);          // different knobs initially
    b.restore(st);
    assert(b.get_mixing_beta() == a.get_mixing_beta());
    assert(b.get_current_step() == a.get_current_step());
    assert(b.get_history_size() == a.get_history_size());

    // Continuing from the restored mixer must match continuing the original:
    // feed the same output to both and compare the next mixed input.
    matrix out = fixed_point_map(x, x_star, 0.4);
    matrix xa = a.mix(out);
    matrix xb = b.mix(out);
    assert(mat_equal(xa, xb, 1e-12));
    std::cout << "[OK] test_snapshot_restore\n";
}

int main(int /*argc*/, char * /*argv*/[])
{
    try {
        test_linear_first_step();
        test_pulay_converges_to_fixed_point();
        test_snapshot_restore();
    } catch (const std::exception &e) {
        std::cerr << "test_pulay_mixer FAILED: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "test_pulay_mixer: all tests passed\n";
    return 0;
}
