// Copyright 2013 Dolphin Emulator Project / 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <compare>
#include <cstdlib>
#include <type_traits>

namespace Common {

template <class T>
struct Rectangle {
    T left{};
    T top{};
    T right{};
    T bottom{};

    constexpr Rectangle() = default;

    constexpr Rectangle(T left, T top, T right, T bottom)
        : left(left), top(top), right(right), bottom(bottom) {}

    constexpr auto operator<=>(const Rectangle&) const = default;

    constexpr void operator*=(const T value) {
        left *= value;
        top *= value;
        right *= value;
        bottom *= value;
    }

    [[nodiscard]] constexpr Rectangle operator*(const T value) const {
        return Rectangle{left * value, top * value, right * value, bottom * value};
    }
    [[nodiscard]] constexpr T GetWidth() const {
        return std::abs(static_cast<std::make_signed_t<T>>(right - left));
    }
    [[nodiscard]] constexpr T GetHeight() const {
        return std::abs(static_cast<std::make_signed_t<T>>(bottom - top));
    }
    [[nodiscard]] constexpr Rectangle<T> TranslateX(const T x) const {
        return Rectangle{left + x, top, right + x, bottom};
    }
    [[nodiscard]] constexpr Rectangle<T> TranslateY(const T y) const {
        return Rectangle{left, top + y, right, bottom + y};
    }
    [[nodiscard]] constexpr Rectangle<T> Scale(const float s) const {
        return Rectangle{left, top, static_cast<T>(left + GetWidth() * s),
                         static_cast<T>(top + GetHeight() * s)};
    }
};

template <typename T>
Rectangle(T, T, T, T) -> Rectangle<T>;

} // namespace Common
