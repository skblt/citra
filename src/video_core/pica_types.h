// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cmath>
#include <cstring>
#include <boost/serialization/access.hpp>
#include "common/common_types.h"

namespace Pica {

/**
 * Template class for converting arbitrary Pica float types to IEEE 754 32-bit single-precision
 * floating point.
 *
 * When decoding, format is as follows:
 *  - The first `M` bits are the mantissa
 *  - The next `E` bits are the exponent
 *  - The last bit is the sign bit
 *
 * @todo Verify on HW if this conversion is sufficiently accurate.
 */
template <u32 M, u32 E>
struct Float {
public:
    [[nodiscard]] constexpr static Float FromFloat32(float val) {
        Float ret;
        ret.value = val;
        return ret;
    }

    [[nodiscard]] constexpr static Float FromRaw(u32 hex) {
        s32 exponent = (hex >> M) & ((1 << E) - 1);
        const s32 width = M + E + 1;
        const s32 bias = 128 - (1 << (E - 1));
        const u32 mantissa = hex & ((1 << M) - 1);
        const u32 sign = (hex >> (E + M)) << 31;

        if (hex & ((1 << (width - 1)) - 1)) {
            if (exponent == (1 << E) - 1)
                exponent = 255;
            else
                exponent += bias;
            hex = sign | (mantissa << (23 - M)) | (exponent << 23);
        } else {
            hex = sign;
        }

        Float result;
        std::memcpy(&result.value, &hex, sizeof(float));
        return result;
    }

    [[nodiscard]] constexpr static Float Zero() {
        return FromFloat32(0.f);
    }

    // Not recommended for anything but logging
    [[nodiscard]] constexpr float ToFloat32() const {
        return value;
    }

    [[nodiscard]] constexpr Float operator*(const Float& flt) const {
        float result = value * flt.ToFloat32();

        // PICA gives 0 instead of NaN when multiplying by inf
        if (std::isnan(result) && !std::isnan(value) && !std::isnan(flt.ToFloat32())) {
            result = 0.f;
        }

        return Float::FromFloat32(result);
    }

    [[nodiscard]] constexpr Float operator/(const Float& flt) const {
        return Float::FromFloat32(ToFloat32() / flt.ToFloat32());
    }

    [[nodiscard]] constexpr Float operator+(const Float& flt) const {
        return Float::FromFloat32(ToFloat32() + flt.ToFloat32());
    }

    [[nodiscard]] constexpr Float operator-(const Float& flt) const {
        return Float::FromFloat32(ToFloat32() - flt.ToFloat32());
    }

    [[nodiscard]] constexpr Float& operator*=(const Float& flt) {
        value = operator*(flt).value;
        return *this;
    }

    [[nodiscard]] constexpr Float& operator/=(const Float& flt) {
        value /= flt.ToFloat32();
        return *this;
    }

    [[nodiscard]] constexpr Float& operator+=(const Float& flt) {
        value += flt.ToFloat32();
        return *this;
    }

    [[nodiscard]] constexpr Float& operator-=(const Float& flt) {
        value -= flt.ToFloat32();
        return *this;
    }

    [[nodiscard]] constexpr Float operator-() const {
        return Float::FromFloat32(-ToFloat32());
    }

    [[nodiscard]] constexpr auto operator<=>(const Float& flt) const {
        return ToFloat32() <=> flt.ToFloat32();
    }

    [[nodiscard]] constexpr bool operator==(const Float& flt) const {
        return ToFloat32() == flt.ToFloat32();
    }

    [[nodiscard]] constexpr bool operator!=(const Float& flt) const {
        return ToFloat32() != flt.ToFloat32();
    }

private:
    static constexpr u32 MASK = (1 << (M + E + 1)) - 1;
    static constexpr u32 MANTISSA_MASK = (1 << M) - 1;
    static constexpr u32 EXPONENT_MASK = (1 << E) - 1;

    // Stored as a regular float, merely for convenience
    // TODO: Perform proper arithmetic on this!
    float value;

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar& value;
    }
};

using float24 = Float<16, 7>;
using float20 = Float<12, 7>;
using float16 = Float<10, 5>;

} // namespace Pica
