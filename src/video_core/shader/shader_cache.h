// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <optional>
#include <unordered_map>
#include <tuple>
#include "video_core/shader/shader.h"

namespace Pica::Shader {

template <typename ShaderType>
using ShaderCacheResult = std::pair<ShaderType, std::optional<std::string>>;

template <typename KeyType, typename ShaderType, auto ModuleCompiler,
          std::string(*CodeGenerator)(const KeyType&)>
class ShaderCache {
public:
    ShaderCache() {}
    ~ShaderCache() = default;

    /// Returns a shader handle generated from the provided config
    template <typename... Args>
    auto Get(const KeyType& config, Args&&... args) -> ShaderCacheResult<ShaderType> {
        auto [iter, new_shader] = shaders.emplace(config, ShaderType{});
        auto& shader = iter->second;

        if (new_shader) {
            std::string code = CodeGenerator(config);
            shader = ModuleCompiler(code, args...);
            return std::make_pair(shader, code);
        }

        return std::make_pair(shader, std::nullopt);
    }

    void Inject(const KeyType& key, ShaderType&& shader) {
        shaders.emplace(key, std::move(shader));
    }

private:
    std::unordered_map<KeyType, ShaderType> shaders;
};

/**
 * This is a cache designed for shaders translated from PICA shaders. The first cache matches the
 * config structure like a normal cache does. On cache miss, the second cache matches the generated
 * GLSL code. The configuration is like this because there might be leftover code in the PICA shader
 * program buffer from the previous shader, which is hashed into the config, resulting several
 * different config values from the same shader program.
 */
template <typename KeyType, typename ShaderType, auto ModuleCompiler,
          std::optional<std::string>(*CodeGenerator)(const Pica::Shader::ShaderSetup&, const KeyType&)>
class ShaderDoubleCache {
public:
    ShaderDoubleCache() = default;
    ~ShaderDoubleCache() = default;

    template <typename... Args>
    auto Get(const KeyType& key, const Pica::Shader::ShaderSetup& setup, Args&&... args) -> ShaderCacheResult<ShaderType> {
        if (auto map_iter = shader_map.find(key); map_iter == shader_map.end()) {
            auto code = CodeGenerator(setup, key);
            if (!code) {
                shader_map[key] = nullptr;
                return std::make_pair(ShaderType{}, std::nullopt);
            }

            std::string& program = code.value();
            auto [iter, new_shader] = shader_cache.emplace(program, ShaderType{});
            auto& shader = iter->second;

            if (new_shader) {
                shader = ModuleCompiler(program, args...);
            }

            shader_map[key] = &shader;
            return std::make_pair(shader, std::move(program));
        } else {
            return std::make_pair(*map_iter->second, std::nullopt);
        }
    }

    void Inject(const KeyType& key, std::string decomp, ShaderType&& program) {
        const auto iter = shader_cache.emplace(std::move(decomp), std::move(program)).first;

        auto& cached_shader = iter->second;
        shader_map.insert_or_assign(key, &cached_shader);
    }

private:
    std::unordered_map<KeyType, ShaderType*> shader_map;
    std::unordered_map<std::string, ShaderType> shader_cache;
};

} // namespace Pica::Shader
