// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/alignment.h"
#include "common/assert.h"
#include "video_core/renderer_opengl/gl_stream_buffer.h"

namespace OpenGL {

StreamBuffer::StreamBuffer(GLenum target, size_t size_)
    : gl_target{target}, buffer_size{size_}, slot_size{buffer_size / SYNC_POINTS},
      buffer_storage{true} {
    for (u64 i = 0; i < SYNC_POINTS; i++) {
        fences[i].Create();
    }

    gl_buffer.Create();
    glBindBuffer(gl_target, gl_buffer.handle);

    if (buffer_storage) {
        glBufferStorage(gl_target, buffer_size, nullptr,
                        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
        mapped_ptr =
            (u8*)glMapBufferRange(gl_target, 0, buffer_size,
                                  GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    } else {
        glBufferData(gl_target, buffer_size, nullptr, GL_STREAM_DRAW);
    }
}

StreamBuffer::~StreamBuffer() {
    if (buffer_storage) {
        glBindBuffer(gl_target, gl_buffer.handle);
        glUnmapBuffer(gl_target);
    }
}

std::tuple<u8*, u64, bool> StreamBuffer::Map(u64 size, u64 alignment) {
    mapped_size = size;

    if (alignment > 0) {
        iterator = Common::AlignUp(iterator, alignment);
    }

    // Insert waiting slots for used memory
    for (u64 i = Slot(used_iterator); i < Slot(iterator); i++) {
        fences[i].Create();
    }
    used_iterator = iterator;

    // Wait for new slots to end of buffer
    for (u64 i = Slot(free_iterator) + 1; i <= Slot(iterator + size) && i < SYNC_POINTS; i++) {
        glClientWaitSync(fences[i].handle, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
        fences[i].Release();
    }

    // If we allocate a large amount of memory (A), commit a smaller amount, then allocate memory
    // smaller than allocation A, we will have already waited for these fences in A, but not used
    // the space. In this case, don't set m_free_iterator to a position before that which we know
    // is safe to use, which would result in waiting on the same fence(s) next time.
    if ((iterator + size) > free_iterator) {
        free_iterator = iterator + size;
    }

    // If buffer is full
    bool invalidate = false;
    if (iterator + size >= buffer_size) {
        invalidate = true;

        // Insert waiting slots in unused space at the end of the buffer
        for (u64 i = Slot(used_iterator); i < SYNC_POINTS; i++) {
            fences[i].Create();
        }

        // Move to the start
        used_iterator = iterator = 0; // offset 0 is always aligned

        // Wait for space at the start
        for (u64 i = 0; i <= Slot(iterator + size); i++) {
            glClientWaitSync(fences[i].handle, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
            fences[i].Release();
        }
        free_iterator = iterator + size;
    }

    u8* pointer{};
    if (buffer_storage) {
        pointer = mapped_ptr + iterator;
    } else {
        pointer = (u8*)glMapBufferRange(gl_target, iterator, size,
                                        GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT |
                                            GL_MAP_UNSYNCHRONIZED_BIT);
    }

    return std::make_tuple(pointer, iterator, invalidate);
}

void StreamBuffer::Unmap(u64 used_size) {
    ASSERT_MSG(used_size <= mapped_size, "Reserved size {} is too small compared to {}",
               mapped_size, used_size);

    if (!buffer_storage) {
        glFlushMappedBufferRange(gl_target, 0, used_size);
        glUnmapBuffer(gl_target);
    }
    iterator += used_size;
}

} // namespace OpenGL
