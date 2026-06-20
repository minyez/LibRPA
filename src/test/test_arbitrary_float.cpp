#include <cassert>
#include <cmath>

#include "../math/arbitrary_float.h"

using Big = librpa_int::ArbitraryBin<128>;
using BigComplex = librpa_int::ArbitraryComplex<128>;

void test_binary_precision()
{
    using Tiny = librpa_int::ArbitraryBin<3>;
    static_assert(Big::precision_bits() == 128);
    assert(Tiny("15").significant_bits() <= Tiny::precision_bits());
    assert(Tiny("15").to_double() == 16.0);
    assert((Big(3) / Big(2)) == Big("1.5"));
    assert((Big(1) / Big(3)).significant_bits() <= Big::precision_bits());
}

void test_integer_arithmetic()
{
    assert((Big("1024") + Big("256")) == Big("1280"));
    assert((Big("123456789") * Big("987654321")) == Big("121932631112635269"));
}

void test_rounded_division()
{
    const double third = (Big(1) / Big(3)).to_double();
    assert(std::abs(third - 1.0 / 3.0) < 1e-15);
}

void test_complex_arithmetic()
{
    const BigComplex x0(Big("2"), Big("2"));
    const BigComplex x(Big("1"), Big("1"));
    const BigComplex value = -Big("1") / (x - x0);
    assert(value.real() == Big("0.5"));
    assert(value.imag() == Big("-0.5"));
}

void test_gmp_compat_layer()
{
    using Mpf = librpa_int::MpfClass<128>;
    using ComplexMpf = librpa_int::ComplexMpf<128>;

    const Mpf one("1", 512);
    const Mpf two(2.0, 512);
    assert(one.get_prec() == 128);
    assert((one / two).get_d() == 0.5);

    const ComplexMpf z(std::complex<double>{1.0, 2.0});
    const ComplexMpf w(Mpf("3"), Mpf("-1"));
    const ComplexMpf product = z * w;
    assert(product.real_mp == Mpf("5"));
    assert(product.imag_mp == Mpf("5"));
    assert(product.abs2() == Mpf("50"));
}

int main()
{
    test_binary_precision();
    test_integer_arithmetic();
    test_rounded_division();
    test_complex_arithmetic();
    test_gmp_compat_layer();
}
