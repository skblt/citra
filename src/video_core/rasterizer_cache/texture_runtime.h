// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <span>
#include <set>
#include "video_core/rasterizer_cache/types.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"

namespace OpenGL {

struct FormatTuple;

struct StagingBuffer {
    u32 size = 0;
    std::span<std::byte> mapped{};
    OGLBuffer buffer{};
    mutable OGLSync buffer_lock{};

    bool operator<(const StagingBuffer& other) const {
        return size < other.size;
    }

    /// Returns true if the buffer does not take part in pending transfer operations
    bool IsFree() const {
        GLint status;
        glGetSynciv(buffer_lock.handle, GL_SYNC_STATUS, 1, nullptr, &status);
        return status == GL_SIGNALED;
    }

    /// Prevents the runtime from reusing the buffer until the transfer operation is complete
    void Lock() const {
        if (buffer_lock) {
            buffer_lock.Release();
        }

        buffer_lock.Create();
    }
};

class Driver;

/**
 * Provides texture manipulation functions to the rasterizer cache
 * Separating this into a class makes it easier to abstract graphics API code
 */
class TextureRuntime {
public:
    TextureRuntime(Driver& driver);
    ~TextureRuntime() = default;

    /// Maps an internal staging buffer of the provided size of pixel uploads/downloads
    const StagingBuffer& FindStaging(u32 size, bool upload);

    /// Copies the GPU pixel data to the provided pixels buffer
    void ReadTexture(OGLTexture& texture, const BufferTextureCopy& copy,
                     PixelFormat format, std::span<std::byte> pixels);

    /// Fills the rectangle of the texture with the clear value provided
    bool ClearTexture(OGLTexture& texture, const TextureClear& clear, ClearValue value);

    /// Copies a rectangle of src_tex to another rectange of dst_rect
    bool CopyTextures(OGLTexture& source, OGLTexture& dest, const TextureCopy& copy);

    /// Blits a rectangle of src_tex to another rectange of dst_rect
    bool BlitTextures(OGLTexture& source, OGLTexture& dest, const TextureBlit& blit);

    /// Generates mipmaps for all the available levels of the texture
    void GenerateMipmaps(OGLTexture& texture, u32 max_level);

private:
    Driver& driver;
    OGLFramebuffer read_fbo, draw_fbo;
    std::multiset<StagingBuffer> upload_buffers;
    std::multiset<StagingBuffer> download_buffers;
};

} // namespace OpenGL
