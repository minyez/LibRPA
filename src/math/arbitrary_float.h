/*!
 * @file arbitrary_float.h
 * @brief Small binary arbitrary-precision floating point type
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace librpa_int
{

using mp_bitcnt_t = std::size_t;

/*!
 * @brief Binary arbitrary-precision float for numerically sensitive algebra.
 *
 * The public precision is in bits. Decimal strings are parsed at the boundary
 * and rounded to this binary precision.
 */
template <std::size_t PrecisionBits = 256>
class ArbitraryBin
{
    static_assert(PrecisionBits > 0, "ArbitraryBin needs at least one precision bit");

public:
    ArbitraryBin() = default;
    ArbitraryBin(const char *value) { *this = parse(value); }
    ArbitraryBin(const char *value, mp_bitcnt_t) : ArbitraryBin(value) {}
    ArbitraryBin(const std::string &value) { *this = parse(value); }
    ArbitraryBin(const std::string &value, mp_bitcnt_t) : ArbitraryBin(value) {}
    ArbitraryBin(int value) : ArbitraryBin(static_cast<long long>(value)) {}
    ArbitraryBin(long value) : ArbitraryBin(static_cast<long long>(value)) {}
    ArbitraryBin(long long value)
    {
        if (value == 0) return;
        unsigned long long magnitude = 0;
        if (value < 0)
        {
            sign_ = -1;
            magnitude = static_cast<unsigned long long>(-(value + 1)) + 1;
        }
        else
        {
            sign_ = 1;
            magnitude = static_cast<unsigned long long>(value);
        }
        limbs_.push_back(static_cast<std::uint32_t>(magnitude));
        if (magnitude >> LIMB_BITS)
            limbs_.push_back(static_cast<std::uint32_t>(magnitude >> LIMB_BITS));
        normalize();
    }
    ArbitraryBin(unsigned int value) : ArbitraryBin(static_cast<unsigned long long>(value)) {}
    ArbitraryBin(unsigned long value) : ArbitraryBin(static_cast<unsigned long long>(value)) {}
    ArbitraryBin(unsigned long long value)
    {
        if (value == 0) return;
        sign_ = 1;
        limbs_.push_back(static_cast<std::uint32_t>(value));
        if (value >> LIMB_BITS) limbs_.push_back(static_cast<std::uint32_t>(value >> LIMB_BITS));
        normalize();
    }
    ArbitraryBin(double value) { assign_floating(value); }
    ArbitraryBin(double value, mp_bitcnt_t) : ArbitraryBin(value) {}
    ArbitraryBin(long double value) { assign_floating(value); }
    ArbitraryBin(long double value, mp_bitcnt_t) : ArbitraryBin(value) {}

    bool is_zero() const noexcept { return sign_ == 0; }
    int sign() const noexcept { return sign_; }
    long long exponent() const noexcept { return exponent_; }
    std::size_t significant_bits() const noexcept { return bit_length(limbs_); }
    static constexpr std::size_t precision_bits() noexcept { return PrecisionBits; }
    mp_bitcnt_t get_prec() const noexcept { return PrecisionBits; }
    double get_d() const { return to_double(); }

    double to_double() const
    {
        if (is_zero()) return 0.0;
        long double value = 0.0L;
        for (std::size_t i = limbs_.size(); i != 0; --i)
            value = std::ldexp(value, LIMB_BITS) + static_cast<long double>(limbs_[i - 1]);

        if (exponent_ > std::numeric_limits<int>::max())
            return sign_ > 0 ? std::numeric_limits<double>::infinity()
                             : -std::numeric_limits<double>::infinity();
        if (exponent_ < std::numeric_limits<int>::min()) return sign_ > 0 ? 0.0 : -0.0;
        return static_cast<double>(std::ldexp(sign_ * value, static_cast<int>(exponent_)));
    }

    std::string to_string() const
    {
        std::ostringstream os;
        os << std::setprecision(std::numeric_limits<long double>::max_digits10)
           << static_cast<long double>(to_double());
        return os.str();
    }

    ArbitraryBin &operator+=(const ArbitraryBin &rhs)
    {
        if (rhs.is_zero()) return *this;
        if (is_zero())
        {
            *this = rhs;
            return *this;
        }

        const long long out_exponent = std::min(exponent_, rhs.exponent_);
        const auto lhs = shift_left(limbs_, static_cast<std::size_t>(exponent_ - out_exponent));
        const auto rhs_limbs =
            shift_left(rhs.limbs_, static_cast<std::size_t>(rhs.exponent_ - out_exponent));

        if (sign_ == rhs.sign_)
        {
            limbs_ = add(lhs, rhs_limbs);
        }
        else
        {
            const int cmp = compare(lhs, rhs_limbs);
            if (cmp == 0)
            {
                *this = ArbitraryBin();
                return *this;
            }
            if (cmp > 0)
            {
                limbs_ = subtract(lhs, rhs_limbs);
            }
            else
            {
                limbs_ = subtract(rhs_limbs, lhs);
                sign_ = rhs.sign_;
            }
        }
        exponent_ = out_exponent;
        normalize();
        return *this;
    }

    ArbitraryBin &operator-=(const ArbitraryBin &rhs)
    {
        ArbitraryBin neg_rhs = rhs;
        neg_rhs.sign_ = -neg_rhs.sign_;
        return *this += neg_rhs;
    }

    ArbitraryBin &operator*=(const ArbitraryBin &rhs)
    {
        if (is_zero() || rhs.is_zero())
        {
            *this = ArbitraryBin();
            return *this;
        }
        limbs_ = multiply(limbs_, rhs.limbs_);
        exponent_ += rhs.exponent_;
        sign_ *= rhs.sign_;
        normalize();
        return *this;
    }

    ArbitraryBin &operator/=(const ArbitraryBin &rhs)
    {
        if (rhs.is_zero()) throw std::domain_error("ArbitraryBin division by zero");
        if (is_zero()) return *this;

        constexpr std::size_t guard_bits = 2;
        const long long shift =
            static_cast<long long>(PrecisionBits + guard_bits + bit_length(rhs.limbs_)) -
            static_cast<long long>(bit_length(limbs_));
        const std::size_t left_shift = shift > 0 ? static_cast<std::size_t>(shift) : 0;

        limbs_ = divide(shift_left(limbs_, left_shift), rhs.limbs_);
        exponent_ -= rhs.exponent_ + static_cast<long long>(left_shift);
        sign_ *= rhs.sign_;
        normalize();
        return *this;
    }

    ArbitraryBin operator-() const
    {
        ArbitraryBin value = *this;
        value.sign_ = -value.sign_;
        return value;
    }

    friend ArbitraryBin operator+(ArbitraryBin lhs, const ArbitraryBin &rhs)
    {
        lhs += rhs;
        return lhs;
    }
    friend ArbitraryBin operator-(ArbitraryBin lhs, const ArbitraryBin &rhs)
    {
        lhs -= rhs;
        return lhs;
    }
    friend ArbitraryBin operator*(ArbitraryBin lhs, const ArbitraryBin &rhs)
    {
        lhs *= rhs;
        return lhs;
    }
    friend ArbitraryBin operator/(ArbitraryBin lhs, const ArbitraryBin &rhs)
    {
        lhs /= rhs;
        return lhs;
    }

    friend bool operator==(const ArbitraryBin &lhs, const ArbitraryBin &rhs)
    {
        return lhs.sign_ == rhs.sign_ && lhs.exponent_ == rhs.exponent_ && lhs.limbs_ == rhs.limbs_;
    }
    friend bool operator!=(const ArbitraryBin &lhs, const ArbitraryBin &rhs)
    {
        return !(lhs == rhs);
    }
    friend bool operator<(const ArbitraryBin &lhs, const ArbitraryBin &rhs)
    {
        if (lhs.sign_ != rhs.sign_) return lhs.sign_ < rhs.sign_;
        if (lhs.sign_ == 0) return false;
        const int cmp = compare_abs(lhs, rhs);
        return lhs.sign_ > 0 ? cmp < 0 : cmp > 0;
    }
    friend bool operator>(const ArbitraryBin &lhs, const ArbitraryBin &rhs) { return rhs < lhs; }
    friend bool operator<=(const ArbitraryBin &lhs, const ArbitraryBin &rhs)
    {
        return !(rhs < lhs);
    }
    friend bool operator>=(const ArbitraryBin &lhs, const ArbitraryBin &rhs)
    {
        return !(lhs < rhs);
    }

    friend ArbitraryBin abs(ArbitraryBin value)
    {
        if (value.sign_ < 0) value.sign_ = 1;
        return value;
    }

    friend std::ostream &operator<<(std::ostream &os, const ArbitraryBin &value)
    {
        return os << value.to_string();
    }

private:
    static constexpr std::size_t LIMB_BITS = 32;
    static constexpr std::uint64_t LIMB_BASE = std::uint64_t{1} << LIMB_BITS;

    int sign_ = 0;
    long long exponent_ = 0;
    // ponytail: simple uint32 limbs; replace with GMP if profiling says this matters.
    std::vector<std::uint32_t> limbs_;

    static ArbitraryBin parse(const std::string &text)
    {
        std::size_t pos = 0;
        while (pos != text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;

        int sign = 1;
        if (pos != text.size() && (text[pos] == '+' || text[pos] == '-'))
        {
            sign = text[pos] == '-' ? -1 : 1;
            ++pos;
        }

        bool seen_digit = false;
        bool seen_decimal = false;
        long long decimal_places = 0;
        std::string digits;
        for (; pos != text.size(); ++pos)
        {
            const char ch = text[pos];
            if (std::isdigit(static_cast<unsigned char>(ch)))
            {
                seen_digit = true;
                digits.push_back(ch);
                if (seen_decimal) ++decimal_places;
                continue;
            }
            if (ch == '.' && !seen_decimal)
            {
                seen_decimal = true;
                continue;
            }
            break;
        }
        if (!seen_digit) throw std::invalid_argument("ArbitraryBin needs digits");

        long long exponent10 = 0;
        if (pos != text.size() && (text[pos] == 'e' || text[pos] == 'E'))
        {
            exponent10 = parse_exponent(text, ++pos);
        }
        else
        {
            while (pos != text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
            if (pos != text.size()) throw std::invalid_argument("Invalid ArbitraryBin literal");
        }

        ArbitraryBin value;
        for (const char ch : digits)
        {
            value *= 10;
            value += ch - '0';
        }

        const long long scale10 = exponent10 - decimal_places;
        for (long long i = 0; i < scale10; ++i) value *= 10;
        for (long long i = 0; i < -scale10; ++i) value /= 10;
        if (sign < 0) value.sign_ = -value.sign_;
        return value;
    }

    static long long parse_exponent(const std::string &text, std::size_t pos)
    {
        while (pos != text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
        const std::size_t begin = pos;
        if (pos != text.size() && (text[pos] == '+' || text[pos] == '-')) ++pos;
        const std::size_t digits_begin = pos;
        while (pos != text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) ++pos;
        if (digits_begin == pos) throw std::invalid_argument("Invalid ArbitraryBin exponent");
        while (pos != text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
        if (pos != text.size()) throw std::invalid_argument("Invalid ArbitraryBin literal");
        return std::stoll(text.substr(begin, pos - begin));
    }

    template <typename T>
    void assign_floating(T value)
    {
        if (!std::isfinite(value)) throw std::invalid_argument("ArbitraryBin needs a finite value");
        std::ostringstream os;
        os << std::setprecision(std::numeric_limits<T>::max_digits10) << value;
        *this = parse(os.str());
    }

    void normalize()
    {
        strip_high_zeroes(limbs_);
        if (limbs_.empty())
        {
            sign_ = 0;
            exponent_ = 0;
            return;
        }

        strip_low_zero_bits(limbs_, exponent_);
        const std::size_t bits = bit_length(limbs_);
        if (bits > PrecisionBits)
        {
            const std::size_t shift = bits - PrecisionBits;
            round_right(limbs_, shift);
            exponent_ += static_cast<long long>(shift);
            if (bit_length(limbs_) > PrecisionBits)
            {
                shift_right(limbs_, 1);
                ++exponent_;
            }
            strip_low_zero_bits(limbs_, exponent_);
        }
    }

    static void strip_high_zeroes(std::vector<std::uint32_t> &limbs)
    {
        while (!limbs.empty() && limbs.back() == 0) limbs.pop_back();
    }

    static int trailing_zero_bits(std::uint32_t value)
    {
        int count = 0;
        while ((value & 1U) == 0U)
        {
            ++count;
            value >>= 1U;
        }
        return count;
    }

    static void strip_low_zero_bits(std::vector<std::uint32_t> &limbs, long long &exponent)
    {
        while (!limbs.empty() && limbs.front() == 0)
        {
            limbs.erase(limbs.begin());
            exponent += LIMB_BITS;
        }
        if (limbs.empty()) return;

        const int shift = trailing_zero_bits(limbs.front());
        if (shift == 0) return;
        shift_right(limbs, static_cast<std::size_t>(shift));
        exponent += shift;
    }

    static std::size_t bit_length(const std::vector<std::uint32_t> &limbs)
    {
        if (limbs.empty()) return 0;
        std::uint32_t top = limbs.back();
        std::size_t bits = (limbs.size() - 1) * LIMB_BITS;
        while (top != 0)
        {
            ++bits;
            top >>= 1U;
        }
        return bits;
    }

    static bool bit_at(const std::vector<std::uint32_t> &limbs, std::size_t bit)
    {
        const std::size_t limb = bit / LIMB_BITS;
        if (limb >= limbs.size()) return false;
        return ((limbs[limb] >> (bit % LIMB_BITS)) & 1U) != 0U;
    }

    static void set_bit(std::vector<std::uint32_t> &limbs, std::size_t bit)
    {
        const std::size_t limb = bit / LIMB_BITS;
        if (limbs.size() <= limb) limbs.resize(limb + 1, 0);
        limbs[limb] |= std::uint32_t{1} << (bit % LIMB_BITS);
    }

    static std::vector<std::uint32_t> shift_left(const std::vector<std::uint32_t> &limbs,
                                                 std::size_t bits)
    {
        if (limbs.empty()) return {};
        const std::size_t whole = bits / LIMB_BITS;
        const std::size_t part = bits % LIMB_BITS;
        std::vector<std::uint32_t> out(whole, 0);
        out.reserve(limbs.size() + whole + 1);

        std::uint64_t carry = 0;
        for (const std::uint32_t limb : limbs)
        {
            const std::uint64_t shifted = (static_cast<std::uint64_t>(limb) << part) | carry;
            out.push_back(static_cast<std::uint32_t>(shifted));
            carry = shifted >> LIMB_BITS;
        }
        if (carry != 0) out.push_back(static_cast<std::uint32_t>(carry));
        strip_high_zeroes(out);
        return out;
    }

    static void shift_right(std::vector<std::uint32_t> &limbs, std::size_t bits)
    {
        const std::size_t whole = bits / LIMB_BITS;
        const std::size_t part = bits % LIMB_BITS;
        if (whole >= limbs.size())
        {
            limbs.clear();
            return;
        }
        limbs.erase(limbs.begin(), limbs.begin() + static_cast<std::ptrdiff_t>(whole));
        if (part != 0)
        {
            std::uint32_t carry = 0;
            for (std::size_t i = limbs.size(); i != 0; --i)
            {
                const std::uint32_t next_carry = limbs[i - 1] << (LIMB_BITS - part);
                limbs[i - 1] = (limbs[i - 1] >> part) | carry;
                carry = next_carry;
            }
        }
        strip_high_zeroes(limbs);
    }

    static void round_right(std::vector<std::uint32_t> &limbs, std::size_t bits)
    {
        if (bits == 0) return;
        const bool round_up = bit_at(limbs, bits - 1);
        shift_right(limbs, bits);
        if (round_up) limbs = add(limbs, {1});
    }

    static int compare(const std::vector<std::uint32_t> &lhs, const std::vector<std::uint32_t> &rhs)
    {
        if (lhs.size() != rhs.size()) return lhs.size() < rhs.size() ? -1 : 1;
        for (std::size_t i = lhs.size(); i != 0; --i)
        {
            if (lhs[i - 1] != rhs[i - 1]) return lhs[i - 1] < rhs[i - 1] ? -1 : 1;
        }
        return 0;
    }

    static int compare_abs(const ArbitraryBin &lhs, const ArbitraryBin &rhs)
    {
        const long long lhs_order = lhs.exponent_ + static_cast<long long>(bit_length(lhs.limbs_));
        const long long rhs_order = rhs.exponent_ + static_cast<long long>(bit_length(rhs.limbs_));
        if (lhs_order != rhs_order) return lhs_order < rhs_order ? -1 : 1;

        const long long out_exponent = std::min(lhs.exponent_, rhs.exponent_);
        return compare(
            shift_left(lhs.limbs_, static_cast<std::size_t>(lhs.exponent_ - out_exponent)),
            shift_left(rhs.limbs_, static_cast<std::size_t>(rhs.exponent_ - out_exponent)));
    }

    static std::vector<std::uint32_t> add(const std::vector<std::uint32_t> &lhs,
                                          const std::vector<std::uint32_t> &rhs)
    {
        std::vector<std::uint32_t> out;
        out.reserve(std::max(lhs.size(), rhs.size()) + 1);
        std::uint64_t carry = 0;
        for (std::size_t i = 0; i != std::max(lhs.size(), rhs.size()) || carry != 0; ++i)
        {
            const std::uint64_t sum =
                carry + (i < lhs.size() ? lhs[i] : 0) + (i < rhs.size() ? rhs[i] : 0);
            out.push_back(static_cast<std::uint32_t>(sum));
            carry = sum >> LIMB_BITS;
        }
        strip_high_zeroes(out);
        return out;
    }

    static std::vector<std::uint32_t> subtract(const std::vector<std::uint32_t> &lhs,
                                               const std::vector<std::uint32_t> &rhs)
    {
        std::vector<std::uint32_t> out;
        out.reserve(lhs.size());
        std::uint64_t borrow = 0;
        for (std::size_t i = 0; i != lhs.size(); ++i)
        {
            const std::uint64_t rhs_limb = (i < rhs.size() ? rhs[i] : 0) + borrow;
            if (static_cast<std::uint64_t>(lhs[i]) < rhs_limb)
            {
                out.push_back(static_cast<std::uint32_t>(LIMB_BASE + lhs[i] - rhs_limb));
                borrow = 1;
            }
            else
            {
                out.push_back(static_cast<std::uint32_t>(lhs[i] - rhs_limb));
                borrow = 0;
            }
        }
        strip_high_zeroes(out);
        return out;
    }

    static std::vector<std::uint32_t> multiply(const std::vector<std::uint32_t> &lhs,
                                               const std::vector<std::uint32_t> &rhs)
    {
        if (lhs.empty() || rhs.empty()) return {};

        std::vector<std::uint32_t> out(lhs.size() + rhs.size() + 1, 0);
        for (std::size_t i = 0; i != lhs.size(); ++i)
        {
            std::uint64_t carry = 0;
            for (std::size_t j = 0; j != rhs.size(); ++j)
            {
                const std::uint64_t total = static_cast<std::uint64_t>(out[i + j]) + carry +
                                            static_cast<std::uint64_t>(lhs[i]) * rhs[j];
                out[i + j] = static_cast<std::uint32_t>(total);
                carry = total >> LIMB_BITS;
            }
            for (std::size_t k = i + rhs.size(); carry != 0; ++k)
            {
                const std::uint64_t total = static_cast<std::uint64_t>(out[k]) + carry;
                out[k] = static_cast<std::uint32_t>(total);
                carry = total >> LIMB_BITS;
            }
        }
        strip_high_zeroes(out);
        return out;
    }

    static void shift_left_one(std::vector<std::uint32_t> &limbs)
    {
        std::uint64_t carry = 0;
        for (std::uint32_t &limb : limbs)
        {
            const std::uint64_t shifted = (static_cast<std::uint64_t>(limb) << 1U) | carry;
            limb = static_cast<std::uint32_t>(shifted);
            carry = shifted >> LIMB_BITS;
        }
        if (carry != 0) limbs.push_back(static_cast<std::uint32_t>(carry));
    }

    static std::vector<std::uint32_t> divide(const std::vector<std::uint32_t> &numerator,
                                             const std::vector<std::uint32_t> &denominator)
    {
        if (denominator.empty()) throw std::domain_error("ArbitraryBin division by zero");

        std::vector<std::uint32_t> quotient;
        std::vector<std::uint32_t> remainder;
        for (std::size_t bit = bit_length(numerator); bit != 0; --bit)
        {
            shift_left_one(remainder);
            if (bit_at(numerator, bit - 1)) set_bit(remainder, 0);
            if (compare(remainder, denominator) >= 0)
            {
                remainder = subtract(remainder, denominator);
                set_bit(quotient, bit - 1);
            }
        }
        strip_high_zeroes(quotient);
        return quotient;
    }
};

template <std::size_t PrecisionBits = 256>
using ArbitraryComplex = std::complex<ArbitraryBin<PrecisionBits>>;

// Below are layers to interface GMP
template <std::size_t PrecisionBits = 256>
using MpfClass = ArbitraryBin<PrecisionBits>;

template <std::size_t PrecisionBits = 256>
class ComplexMpf
{
public:
    MpfClass<PrecisionBits> real_mp;
    MpfClass<PrecisionBits> imag_mp;

    ComplexMpf() = default;
    ComplexMpf(const MpfClass<PrecisionBits> &real_mp, const MpfClass<PrecisionBits> &imag_mp)
        : real_mp(real_mp), imag_mp(imag_mp)
    {
    }
    ComplexMpf(std::complex<double> value) : real_mp(value.real()), imag_mp(value.imag()) {}

    ComplexMpf operator+(const ComplexMpf &rhs) const
    {
        return {real_mp + rhs.real_mp, imag_mp + rhs.imag_mp};
    }
    ComplexMpf operator-(const ComplexMpf &rhs) const
    {
        return {real_mp - rhs.real_mp, imag_mp - rhs.imag_mp};
    }
    ComplexMpf operator*(const ComplexMpf &rhs) const
    {
        return {real_mp * rhs.real_mp - imag_mp * rhs.imag_mp,
                real_mp * rhs.imag_mp + imag_mp * rhs.real_mp};
    }
    ComplexMpf operator/(const ComplexMpf &rhs) const
    {
        const auto denominator = rhs.real_mp * rhs.real_mp + rhs.imag_mp * rhs.imag_mp;
        return {(real_mp * rhs.real_mp + imag_mp * rhs.imag_mp) / denominator,
                (rhs.real_mp * imag_mp - real_mp * rhs.imag_mp) / denominator};
    }

    MpfClass<PrecisionBits> abs2() const { return real_mp * real_mp + imag_mp * imag_mp; }

    operator std::complex<double>() const { return {real_mp.get_d(), imag_mp.get_d()}; }
};

template <std::size_t PrecisionBits = 256>
using GmpFloat = MpfClass<PrecisionBits>;

template <std::size_t PrecisionBits = 256>
using GmpComplex = ComplexMpf<PrecisionBits>;

}  // namespace librpa_int
