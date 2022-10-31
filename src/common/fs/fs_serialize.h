// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <functional>
#include <filesystem>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/wrapper.hpp>

namespace Common::FS {

// Replaces install-specific paths with standard placeholders, and back again
std::filesystem::path SerializePath(const std::filesystem::path& input, bool is_saving);

// A serializable path string
struct Path : public boost::serialization::wrapper_traits<const Path> {
    std::filesystem::path& path;

    explicit Path(std::filesystem::path& path) : path{path} {}

    static const Path make(std::filesystem::path& path) {
        return Path(path);
    }

    template <class Archive>
    void save(Archive& ar, const unsigned int) const {
        auto s_path = SerializePath(path, true);
        ar << s_path;
    }
    template <class Archive>
    void load(Archive& ar, const unsigned int) const {
        ar >> path;
        path = SerializePath(path, false);
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER();
    friend class boost::serialization::access;
};

} // namespace Common::FS
