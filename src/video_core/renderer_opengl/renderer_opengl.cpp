// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <queue>
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/dumping/backend.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/hw/hw.h"
#include "core/hw/lcd.h"
#include "core/memory.h"
#include "core/tracer/recorder.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/renderer_opengl/gl_rasterizer.h"
#include "video_core/renderer_opengl/gl_shader_util.h"
#include "video_core/renderer_opengl/gl_state.h"
#include "video_core/renderer_opengl/gl_vars.h"
#include "video_core/renderer_opengl/post_processing_opengl.h"
#include "video_core/renderer_opengl/renderer_opengl.h"
#include "video_core/video_core.h"

#include "video_core/host_shaders/opengl_present_anaglyph_frag.h"
#include "video_core/host_shaders/opengl_present_frag.h"
#include "video_core/host_shaders/opengl_present_interlaced_frag.h"
#include "video_core/host_shaders/opengl_present_vert.h"

MICROPROFILE_DEFINE(OpenGL_RenderFrame, "OpenGL", "Render Frame", MP_RGB(128, 128, 64));
MICROPROFILE_DEFINE(OpenGL_WaitPresent, "OpenGL", "Wait For Present", MP_RGB(128, 128, 128));

namespace OpenGL {

// If the size of this is too small, it ends up creating a soft cap on FPS as the renderer will have
// to wait on available presentation frames. There doesn't seem to be much of a downside to a larger
// number but 9 swap textures at 60FPS presentation allows for 800% speed so thats probably fine
#ifdef ANDROID
// Reduce the size of swap_chain, since the UI only allows upto 200% speed.
constexpr std::size_t SWAP_CHAIN_SIZE = 6;
#else
constexpr std::size_t SWAP_CHAIN_SIZE = 9;
#endif

class OGLTextureMailboxException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class OGLTextureMailbox : public Frontend::TextureMailbox {
public:
    std::mutex swap_chain_lock;
    std::condition_variable free_cv;
    std::condition_variable present_cv;
    std::array<Frontend::Frame, SWAP_CHAIN_SIZE> swap_chain{};
    std::queue<Frontend::Frame*> free_queue{};
    std::deque<Frontend::Frame*> present_queue{};
    Frontend::Frame* previous_frame = nullptr;

    OGLTextureMailbox() {
        for (auto& frame : swap_chain) {
            free_queue.push(&frame);
        }
    }

    ~OGLTextureMailbox() override {
        // lock the mutex and clear out the present and free_queues and notify any people who are
        // blocked to prevent deadlock on shutdown
        std::scoped_lock lock(swap_chain_lock);
        std::queue<Frontend::Frame*>().swap(free_queue);
        present_queue.clear();
        present_cv.notify_all();
        free_cv.notify_all();
    }

    void ReloadPresentFrame(Frontend::Frame* frame, u32 height, u32 width) override {
        frame->present.Release();
        frame->present.Create();
        GLint previous_draw_fbo{};
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previous_draw_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, frame->present.handle);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                  frame->color.handle);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_CRITICAL(Render_OpenGL, "Failed to recreate present FBO!");
        }
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previous_draw_fbo);
        frame->color_reloaded = false;
    }

    void ReloadRenderFrame(Frontend::Frame* frame, u32 width, u32 height) override {
        OpenGLState prev_state = OpenGLState::GetCurState();
        OpenGLState state = OpenGLState::GetCurState();

        // Recreate the color texture attachment
        frame->color.Release();
        frame->color.Create();
        state.renderbuffer = frame->color.handle;
        state.Apply();
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);

        // Recreate the FBO for the render target
        frame->render.Release();
        frame->render.Create();
        state.draw.read_framebuffer = frame->render.handle;
        state.draw.draw_framebuffer = frame->render.handle;
        state.Apply();
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                  frame->color.handle);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_CRITICAL(Render_OpenGL, "Failed to recreate render FBO!");
        }
        prev_state.Apply();
        frame->width = width;
        frame->height = height;
        frame->color_reloaded = true;
    }

    Frontend::Frame* GetRenderFrame() override {
        std::unique_lock<std::mutex> lock(swap_chain_lock);

        // If theres no free frames, we will reuse the oldest render frame
        if (free_queue.empty()) {
            auto frame = present_queue.back();
            present_queue.pop_back();
            return frame;
        }

        Frontend::Frame* frame = free_queue.front();
        free_queue.pop();
        return frame;
    }

    void ReleaseRenderFrame(Frontend::Frame* frame) override {
        std::unique_lock<std::mutex> lock(swap_chain_lock);
        present_queue.push_front(frame);
        present_cv.notify_one();
    }

    // This is virtual as it is to be overriden in OGLVideoDumpingMailbox below.
    virtual void LoadPresentFrame() {
        // free the previous frame and add it back to the free queue
        if (previous_frame) {
            free_queue.push(previous_frame);
            free_cv.notify_one();
        }

        // the newest entries are pushed to the front of the queue
        Frontend::Frame* frame = present_queue.front();
        present_queue.pop_front();
        // remove all old entries from the present queue and move them back to the free_queue
        for (auto f : present_queue) {
            free_queue.push(f);
        }
        present_queue.clear();
        previous_frame = frame;
    }

    Frontend::Frame* TryGetPresentFrame(int timeout_ms) override {
        std::unique_lock<std::mutex> lock(swap_chain_lock);
        // wait for new entries in the present_queue
        present_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            [&] { return !present_queue.empty(); });
        if (present_queue.empty()) {
            // timed out waiting for a frame to draw so return the previous frame
            return previous_frame;
        }

        LoadPresentFrame();
        return previous_frame;
    }
};

/// This mailbox is different in that it will never discard rendered frames
class OGLVideoDumpingMailbox : public OGLTextureMailbox {
public:
    bool quit = false;

    Frontend::Frame* GetRenderFrame() override {
        std::unique_lock<std::mutex> lock(swap_chain_lock);

        // If theres no free frames, we will wait until one shows up
        if (free_queue.empty()) {
            free_cv.wait(lock, [&] { return (!free_queue.empty() || quit); });
            if (quit) {
                throw OGLTextureMailboxException("VideoDumpingMailbox quitting");
            }

            if (free_queue.empty()) {
                LOG_CRITICAL(Render_OpenGL, "Could not get free frame");
                return nullptr;
            }
        }

        Frontend::Frame* frame = free_queue.front();
        free_queue.pop();
        return frame;
    }

    void LoadPresentFrame() override {
        // free the previous frame and add it back to the free queue
        if (previous_frame) {
            free_queue.push(previous_frame);
            free_cv.notify_one();
        }

        Frontend::Frame* frame = present_queue.back();
        present_queue.pop_back();
        previous_frame = frame;

        // Do not remove entries from the present_queue, as video dumping would require
        // that we preserve all frames
    }

    Frontend::Frame* TryGetPresentFrame(int timeout_ms) override {
        std::unique_lock<std::mutex> lock(swap_chain_lock);
        // wait for new entries in the present_queue
        present_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            [&] { return !present_queue.empty(); });
        if (present_queue.empty()) {
            // timed out waiting for a frame
            return nullptr;
        }

        LoadPresentFrame();
        return previous_frame;
    }
};

/**
 * Vertex structure that the drawn screen rectangles are composed of.
 */
struct ScreenRectVertex {
    ScreenRectVertex(GLfloat x, GLfloat y, GLfloat u, GLfloat v) {
        position[0] = x;
        position[1] = y;
        tex_coord[0] = u;
        tex_coord[1] = v;
    }

    GLfloat position[2];
    GLfloat tex_coord[2];
};

/**
 * Defines a 1:1 pixel ortographic projection matrix with (0,0) on the top-left
 * corner and (width, height) on the lower-bottom.
 *
 * The projection part of the matrix is trivial, hence these operations are represented
 * by a 3x2 matrix.
 *
 * @param flipped Whether the frame should be flipped upside down.
 */
static std::array<GLfloat, 3 * 2> MakeOrthographicMatrix(const float width, const float height,
                                                         bool flipped) {

    std::array<GLfloat, 3 * 2> matrix; // Laid out in column-major order

    // Last matrix row is implicitly assumed to be [0, 0, 1].
    if (flipped) {
        // clang-format off
        matrix[0] = 2.f / width; matrix[2] = 0.f;           matrix[4] = -1.f;
        matrix[1] = 0.f;         matrix[3] = 2.f / height;  matrix[5] = -1.f;
        // clang-format on
    } else {
        // clang-format off
        matrix[0] = 2.f / width; matrix[2] = 0.f;           matrix[4] = -1.f;
        matrix[1] = 0.f;         matrix[3] = -2.f / height; matrix[5] = 1.f;
        // clang-format on
    }

    return matrix;
}

RendererOpenGL::RendererOpenGL(Core::System& system_, Frontend::EmuWindow& window,
                               Frontend::EmuWindow* secondary_window)
    : RendererBase{window, secondary_window}, system{system_}, memory{system.Memory()},
      driver{Settings::values.graphics_api.GetValue() == Settings::GraphicsAPI::OpenGLES,
             Settings::values.renderer_debug.GetValue()},
      rasterizer{memory, system.CustomTexManager(), render_window, driver},
      frame_dumper{system.VideoDumper(), window} {

    const Vendor vendor = driver.GetVendor();
    if (vendor == Vendor::Generic || vendor == Vendor::Unknown) {
        LOG_WARNING(Render_OpenGL, "Unknown vendor: {}", driver.GetVendorString());
    }

    InitOpenGLObjects();

    window.mailbox = std::make_unique<OGLTextureMailbox>();
    if (secondary_window) {
        secondary_window->mailbox = std::make_unique<OGLTextureMailbox>();
    }
    frame_dumper.mailbox = std::make_unique<OGLVideoDumpingMailbox>();
}

RendererOpenGL::~RendererOpenGL() = default;

void RendererOpenGL::SwapBuffers() {
    // Maintain the rasterizer's state as a priority
    OpenGLState prev_state = OpenGLState::GetCurState();
    state.Apply();

    PrepareRendertarget();
    RenderScreenshot();

    const auto& main_layout = render_window.GetFramebufferLayout();
    RenderToMailbox(main_layout, render_window.mailbox, false);

#ifndef ANDROID
    if (Settings::values.layout_option.GetValue() == Settings::LayoutOption::SeparateWindows) {
        ASSERT(secondary_window);
        const auto& secondary_layout = secondary_window->GetFramebufferLayout();
        RenderToMailbox(secondary_layout, secondary_window->mailbox, false);
        secondary_window->PollEvents();
    }
#endif
    if (frame_dumper.IsDumping()) {
        try {
            RenderToMailbox(frame_dumper.GetLayout(), frame_dumper.mailbox, true);
        } catch (const OGLTextureMailboxException& exception) {
            LOG_DEBUG(Render_OpenGL, "Frame dumper exception caught: {}", exception.what());
        }
    }

    m_current_frame++;
    system.perf_stats->EndSystemFrame();

    render_window.PollEvents();

    system.frame_limiter.DoFrameLimiting(system.CoreTiming().GetGlobalTimeUs());
    system.perf_stats->BeginSystemFrame();

    prev_state.Apply();

    if (Pica::g_debug_context && Pica::g_debug_context->recorder) {
        Pica::g_debug_context->recorder->FrameFinished();
    }
}

void RendererOpenGL::RenderScreenshot() {
    if (settings.screenshot_requested.exchange(false)) {
        // Draw this frame to the screenshot framebuffer
        screenshot_framebuffer.Create();
        GLuint old_read_fb = state.draw.read_framebuffer;
        GLuint old_draw_fb = state.draw.draw_framebuffer;
        state.draw.read_framebuffer = state.draw.draw_framebuffer = screenshot_framebuffer.handle;
        state.Apply();

        const Layout::FramebufferLayout layout{settings.screenshot_framebuffer_layout};

        GLuint renderbuffer;
        glGenRenderbuffers(1, &renderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, layout.width, layout.height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                  renderbuffer);

        DrawScreens(layout, false);

        glReadPixels(0, 0, layout.width, layout.height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                     settings.screenshot_bits);

        screenshot_framebuffer.Release();
        state.draw.read_framebuffer = old_read_fb;
        state.draw.draw_framebuffer = old_draw_fb;
        state.Apply();
        glDeleteRenderbuffers(1, &renderbuffer);

        settings.screenshot_complete_callback();
    }
}

void RendererOpenGL::PrepareRendertarget() {
    for (int i : {0, 1, 2}) {
        int fb_id = i == 2 ? 1 : 0;
        const auto& framebuffer = GPU::g_regs.framebuffer_config[fb_id];

        // Main LCD (0): 0x1ED02204, Sub LCD (1): 0x1ED02A04
        u32 lcd_color_addr =
            (fb_id == 0) ? LCD_REG_INDEX(color_fill_top) : LCD_REG_INDEX(color_fill_bottom);
        lcd_color_addr = HW::VADDR_LCD + 4 * lcd_color_addr;
        LCD::Regs::ColorFill color_fill = {0};
        LCD::Read(color_fill.raw, lcd_color_addr);

        if (color_fill.is_enabled) {
            LoadColorToActiveGLTexture(color_fill.color_r, color_fill.color_g, color_fill.color_b,
                                       screen_infos[i].texture);

            // Resize the texture in case the framebuffer size has changed
            screen_infos[i].texture.width = 1;
            screen_infos[i].texture.height = 1;
        } else {
            if (screen_infos[i].texture.width != (GLsizei)framebuffer.width ||
                screen_infos[i].texture.height != (GLsizei)framebuffer.height ||
                screen_infos[i].texture.format != framebuffer.color_format) {
                // Reallocate texture if the framebuffer size has changed.
                // This is expected to not happen very often and hence should not be a
                // performance problem.
                ConfigureFramebufferTexture(screen_infos[i].texture, framebuffer);
            }
            LoadFBToScreenInfo(framebuffer, screen_infos[i], i == 1);

            // Resize the texture in case the framebuffer size has changed
            screen_infos[i].texture.width = framebuffer.width;
            screen_infos[i].texture.height = framebuffer.height;
        }
    }
}

void RendererOpenGL::RenderToMailbox(const Layout::FramebufferLayout& layout,
                                     std::unique_ptr<Frontend::TextureMailbox>& mailbox,
                                     bool flipped) {

    Frontend::Frame* frame;
    {
        MICROPROFILE_SCOPE(OpenGL_WaitPresent);

        frame = mailbox->GetRenderFrame();

        // Clean up sync objects before drawing

        // INTEL driver workaround. We can't delete the previous render sync object until we are
        // sure that the presentation is done
        if (frame->present_fence.handle) {
            glClientWaitSync(frame->present_fence.handle, 0, GL_TIMEOUT_IGNORED);
        }

        // delete the draw fence if the frame wasn't presented
        frame->render_fence.Release();

        // wait for the presentation to be done
        if (frame->present_fence.handle) {
            glWaitSync(frame->present_fence.handle, 0, GL_TIMEOUT_IGNORED);
            frame->present_fence.Release();
        }
    }

    {
        MICROPROFILE_SCOPE(OpenGL_RenderFrame);
        // Recreate the frame if the size of the window has changed
        if (layout.width != frame->width || layout.height != frame->height) {
            LOG_DEBUG(Render_OpenGL, "Reloading render frame");
            mailbox->ReloadRenderFrame(frame, layout.width, layout.height);
        }

        state.draw.draw_framebuffer = frame->render.handle;
        state.Apply();
        DrawScreens(layout, flipped);
        // Create a fence for the frontend to wait on and swap this frame to OffTex
        frame->render_fence.Create();
        glFlush();
        mailbox->ReleaseRenderFrame(frame);
    }
}

void RendererOpenGL::LoadFBToScreenInfo(const GPU::Regs::FramebufferConfig& framebuffer,
                                        ScreenInfo& screen_info, bool right_eye) {

    if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0)
        right_eye = false;

    const PAddr framebuffer_addr =
        framebuffer.active_fb == 0
            ? (!right_eye ? framebuffer.address_left1 : framebuffer.address_right1)
            : (!right_eye ? framebuffer.address_left2 : framebuffer.address_right2);

    LOG_TRACE(Render_OpenGL, "0x{:08x} bytes from 0x{:08x}({}x{}), fmt {:x}",
              framebuffer.stride * framebuffer.height, framebuffer_addr, framebuffer.width.Value(),
              framebuffer.height.Value(), framebuffer.format);

    int bpp = GPU::Regs::BytesPerPixel(framebuffer.color_format);
    std::size_t pixel_stride = framebuffer.stride / bpp;

    // OpenGL only supports specifying a stride in units of pixels, not bytes, unfortunately
    ASSERT(pixel_stride * bpp == framebuffer.stride);

    // Ensure no bad interactions with GL_UNPACK_ALIGNMENT, which by default
    // only allows rows to have a memory alignement of 4.
    ASSERT(pixel_stride % 4 == 0);

    if (!rasterizer.AccelerateDisplay(framebuffer, framebuffer_addr, static_cast<u32>(pixel_stride),
                                      screen_info)) {
        // Reset the screen info's display texture to its own permanent texture
        screen_info.display_texture = screen_info.texture.resource.handle;
        screen_info.display_texcoords = Common::Rectangle<float>(0.f, 0.f, 1.f, 1.f);

        Memory::RasterizerFlushRegion(framebuffer_addr, framebuffer.stride * framebuffer.height);
        const u8* framebuffer_data = memory.GetPhysicalPointer(framebuffer_addr);

        state.texture_units[0].texture_2d = screen_info.texture.resource.handle;
        state.Apply();

        glActiveTexture(GL_TEXTURE0);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, (GLint)pixel_stride);

        // Update existing texture
        // TODO: Test what happens on hardware when you change the framebuffer dimensions so that
        //       they differ from the LCD resolution.
        // TODO: Applications could theoretically crash Citra here by specifying too large
        //       framebuffer sizes. We should make sure that this cannot happen.
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, framebuffer.width, framebuffer.height,
                        screen_info.texture.gl_format, screen_info.texture.gl_type,
                        framebuffer_data);

        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

        state.texture_units[0].texture_2d = 0;
        state.Apply();
    }
}

void RendererOpenGL::LoadColorToActiveGLTexture(u8 color_r, u8 color_g, u8 color_b,
                                                const TextureInfo& texture) {
    state.texture_units[0].texture_2d = texture.resource.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    u8 framebuffer_data[3] = {color_r, color_g, color_b};

    // Update existing texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, framebuffer_data);

    state.texture_units[0].texture_2d = 0;
    state.Apply();
}

void RendererOpenGL::InitOpenGLObjects() {
    glClearColor(Settings::values.bg_red.GetValue(), Settings::values.bg_green.GetValue(),
                 Settings::values.bg_blue.GetValue(), 0.0f);

    // Configure present samplers
    for (std::size_t i = 0; i < present_samplers.size(); i++) {
        const GLint filter = i == 0 ? GL_LINEAR : GL_NEAREST;
        OGLSampler& sampler = present_samplers[i];
        sampler.Create();
        glSamplerParameteri(sampler.handle, GL_TEXTURE_MIN_FILTER, filter);
        glSamplerParameteri(sampler.handle, GL_TEXTURE_MAG_FILTER, filter);
        glSamplerParameteri(sampler.handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(sampler.handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    ReloadShader();

    vertex_buffer.Create();
    vertex_array.Create();

    state.draw.vertex_array = vertex_array.handle;
    state.draw.vertex_buffer = vertex_buffer.handle;
    state.draw.uniform_buffer = 0;
    state.Apply();

    // Attach vertex data to VAO
    glBufferData(GL_ARRAY_BUFFER, sizeof(ScreenRectVertex) * 4, nullptr, GL_STREAM_DRAW);
    glVertexAttribPointer(attrib_position, 2, GL_FLOAT, GL_FALSE, sizeof(ScreenRectVertex),
                          (GLvoid*)offsetof(ScreenRectVertex, position));
    glVertexAttribPointer(attrib_tex_coord, 2, GL_FLOAT, GL_FALSE, sizeof(ScreenRectVertex),
                          (GLvoid*)offsetof(ScreenRectVertex, tex_coord));
    glEnableVertexAttribArray(attrib_position);
    glEnableVertexAttribArray(attrib_tex_coord);

    // Allocate textures for each screen
    for (auto& screen_info : screen_infos) {
        screen_info.texture.resource.Create();

        // Allocation of storage is deferred until the first frame, when we
        // know the framebuffer size.
        state.texture_units[0].texture_2d = screen_info.texture.resource.handle;
        state.Apply();

        glActiveTexture(GL_TEXTURE0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        screen_info.display_texture = screen_info.texture.resource.handle;
    }

    state.texture_units[0].texture_2d = 0;
    state.Apply();
}

void RendererOpenGL::ReloadSampler() {
    current_sampler = !Settings::values.filter_mode.GetValue();
}

void RendererOpenGL::ReloadShader() {
    std::string shader_data;
    if (GLES) {
        shader_data += fragment_shader_precision_OES;
    }

    const Settings::StereoRenderOption render_3d = Settings::values.render_3d.GetValue();
    if (render_3d == Settings::StereoRenderOption::Anaglyph) {
        if (Settings::values.anaglyph_shader_name.GetValue() == "dubois (builtin)") {
            shader_data += HostShaders::OPENGL_PRESENT_ANAGLYPH_FRAG;
        } else {
            std::string shader_text = OpenGL::GetPostProcessingShaderCode(
                true, Settings::values.anaglyph_shader_name.GetValue());
            if (shader_text.empty()) {
                // Should probably provide some information that the shader couldn't load
                shader_data += HostShaders::OPENGL_PRESENT_ANAGLYPH_FRAG;
            } else {
                shader_data += shader_text;
            }
        }
    } else if (render_3d == Settings::StereoRenderOption::Interlaced ||
               render_3d == Settings::StereoRenderOption::ReverseInterlaced) {
        shader_data += HostShaders::OPENGL_PRESENT_INTERLACED_FRAG;
    } else {
        if (Settings::values.pp_shader_name.GetValue() == "none (builtin)") {
            shader_data += HostShaders::OPENGL_PRESENT_INTERLACED_FRAG;
        } else {
            std::string shader_text = OpenGL::GetPostProcessingShaderCode(
                false, Settings::values.pp_shader_name.GetValue());
            if (shader_text.empty()) {
                // Should probably provide some information that the shader couldn't load
                shader_data += HostShaders::OPENGL_PRESENT_INTERLACED_FRAG;
            } else {
                shader_data += shader_text;
            }
        }
    }
    shader.Create(HostShaders::OPENGL_PRESENT_VERT, shader_data.c_str());
    state.draw.shader_program = shader.handle;
    state.Apply();
    uniform_modelview_matrix = glGetUniformLocation(shader.handle, "modelview_matrix");
    uniform_color_texture = glGetUniformLocation(shader.handle, "color_texture");
    if (render_3d == Settings::StereoRenderOption::Anaglyph ||
        render_3d == Settings::StereoRenderOption::Interlaced ||
        render_3d == Settings::StereoRenderOption::ReverseInterlaced) {
        uniform_color_texture_r = glGetUniformLocation(shader.handle, "color_texture_r");
    }
    if (render_3d == Settings::StereoRenderOption::Interlaced ||
        render_3d == Settings::StereoRenderOption::ReverseInterlaced) {
        const GLuint uniform_reverse_interlaced =
            glGetUniformLocation(shader.handle, "reverse_interlaced");
        if (render_3d == Settings::StereoRenderOption::ReverseInterlaced)
            glUniform1i(uniform_reverse_interlaced, 1);
        else
            glUniform1i(uniform_reverse_interlaced, 0);
    }
    uniform_i_resolution = glGetUniformLocation(shader.handle, "i_resolution");
    uniform_o_resolution = glGetUniformLocation(shader.handle, "o_resolution");
    uniform_layer = glGetUniformLocation(shader.handle, "layer");
    attrib_position = glGetAttribLocation(shader.handle, "vert_position");
    attrib_tex_coord = glGetAttribLocation(shader.handle, "vert_tex_coord");
}

void RendererOpenGL::ConfigureFramebufferTexture(TextureInfo& texture,
                                                 const GPU::Regs::FramebufferConfig& framebuffer) {
    GPU::Regs::PixelFormat format = framebuffer.color_format;
    GLint internal_format{};

    texture.format = format;
    texture.width = framebuffer.width;
    texture.height = framebuffer.height;

    switch (format) {
    case GPU::Regs::PixelFormat::RGBA8:
        internal_format = GL_RGBA;
        texture.gl_format = GL_RGBA;
        texture.gl_type = GLES ? GL_UNSIGNED_BYTE : GL_UNSIGNED_INT_8_8_8_8;
        break;

    case GPU::Regs::PixelFormat::RGB8:
        // This pixel format uses BGR since GL_UNSIGNED_BYTE specifies byte-order, unlike every
        // specific OpenGL type used in this function using native-endian (that is, little-endian
        // mostly everywhere) for words or half-words.
        // TODO: check how those behave on big-endian processors.
        internal_format = GL_RGB;

        // GLES Dosen't support BGR , Use RGB instead
        texture.gl_format = GLES ? GL_RGB : GL_BGR;
        texture.gl_type = GL_UNSIGNED_BYTE;
        break;

    case GPU::Regs::PixelFormat::RGB565:
        internal_format = GL_RGB;
        texture.gl_format = GL_RGB;
        texture.gl_type = GL_UNSIGNED_SHORT_5_6_5;
        break;

    case GPU::Regs::PixelFormat::RGB5A1:
        internal_format = GL_RGBA;
        texture.gl_format = GL_RGBA;
        texture.gl_type = GL_UNSIGNED_SHORT_5_5_5_1;
        break;

    case GPU::Regs::PixelFormat::RGBA4:
        internal_format = GL_RGBA;
        texture.gl_format = GL_RGBA;
        texture.gl_type = GL_UNSIGNED_SHORT_4_4_4_4;
        break;

    default:
        UNIMPLEMENTED();
    }

    state.texture_units[0].texture_2d = texture.resource.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, texture.width, texture.height, 0,
                 texture.gl_format, texture.gl_type, nullptr);

    state.texture_units[0].texture_2d = 0;
    state.Apply();
}

void RendererOpenGL::DrawSingleScreenRotated(const ScreenInfo& screen_info, float x, float y,
                                             float w, float h) {
    const auto& texcoords = screen_info.display_texcoords;

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
    }};

    // As this is the "DrawSingleScreenRotated" function, the output resolution dimensions have been
    // swapped. If a non-rotated draw-screen function were to be added for book-mode games, those
    // should probably be set to the standard (w, h, 1.0 / w, 1.0 / h) ordering.
    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    glUniform4f(uniform_i_resolution, static_cast<float>(screen_info.texture.width * scale_factor),
                static_cast<float>(screen_info.texture.height * scale_factor),
                1.0f / static_cast<float>(screen_info.texture.width * scale_factor),
                1.0f / static_cast<float>(screen_info.texture.height * scale_factor));
    glUniform4f(uniform_o_resolution, h, w, 1.0f / h, 1.0f / w);
    state.texture_units[0].texture_2d = screen_info.display_texture;
    state.texture_units[0].sampler = present_samplers[current_sampler].handle;
    state.Apply();

    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    state.texture_units[0].texture_2d = 0;
    state.texture_units[0].sampler = 0;
    state.Apply();
}

void RendererOpenGL::DrawSingleScreen(const ScreenInfo& screen_info, float x, float y, float w,
                                      float h) {
    const auto& texcoords = screen_info.display_texcoords;

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
    }};

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    glUniform4f(uniform_i_resolution, static_cast<float>(screen_info.texture.width * scale_factor),
                static_cast<float>(screen_info.texture.height * scale_factor),
                1.0f / static_cast<float>(screen_info.texture.width * scale_factor),
                1.0f / static_cast<float>(screen_info.texture.height * scale_factor));
    glUniform4f(uniform_o_resolution, w, h, 1.0f / w, 1.0f / h);
    state.texture_units[0].texture_2d = screen_info.display_texture;
    state.texture_units[0].sampler = present_samplers[current_sampler].handle;
    state.Apply();

    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    state.texture_units[0].texture_2d = 0;
    state.texture_units[0].sampler = 0;
    state.Apply();
}

void RendererOpenGL::DrawSingleScreenStereoRotated(const ScreenInfo& screen_info_l,
                                                   const ScreenInfo& screen_info_r, float x,
                                                   float y, float w, float h) {
    const auto& texcoords = screen_info_l.display_texcoords;

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
    }};

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    glUniform4f(uniform_i_resolution,
                static_cast<float>(screen_info_l.texture.width * scale_factor),
                static_cast<float>(screen_info_l.texture.height * scale_factor),
                1.0f / static_cast<float>(screen_info_l.texture.width * scale_factor),
                1.0f / static_cast<float>(screen_info_l.texture.height * scale_factor));
    glUniform4f(uniform_o_resolution, h, w, 1.0f / h, 1.0f / w);
    state.texture_units[0].texture_2d = screen_info_l.display_texture;
    state.texture_units[1].texture_2d = screen_info_r.display_texture;
    state.texture_units[0].sampler = present_samplers[current_sampler].handle;
    state.texture_units[1].sampler = present_samplers[current_sampler].handle;
    state.Apply();

    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    state.texture_units[0].texture_2d = 0;
    state.texture_units[1].texture_2d = 0;
    state.texture_units[0].sampler = 0;
    state.texture_units[1].sampler = 0;
    state.Apply();
}

void RendererOpenGL::DrawSingleScreenStereo(const ScreenInfo& screen_info_l,
                                            const ScreenInfo& screen_info_r, float x, float y,
                                            float w, float h) {
    const auto& texcoords = screen_info_l.display_texcoords;

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
    }};

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    glUniform4f(uniform_i_resolution,
                static_cast<float>(screen_info_l.texture.width * scale_factor),
                static_cast<float>(screen_info_l.texture.height * scale_factor),
                1.0f / static_cast<float>(screen_info_l.texture.width * scale_factor),
                1.0f / static_cast<float>(screen_info_l.texture.height * scale_factor));
    glUniform4f(uniform_o_resolution, w, h, 1.0f / w, 1.0f / h);
    state.texture_units[0].texture_2d = screen_info_l.display_texture;
    state.texture_units[1].texture_2d = screen_info_r.display_texture;
    state.texture_units[0].sampler = present_samplers[current_sampler].handle;
    state.texture_units[1].sampler = present_samplers[current_sampler].handle;
    state.Apply();

    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    state.texture_units[0].texture_2d = 0;
    state.texture_units[1].texture_2d = 0;
    state.texture_units[0].sampler = 0;
    state.texture_units[1].sampler = 0;
    state.Apply();
}

void RendererOpenGL::DrawScreens(const Layout::FramebufferLayout& layout, bool flipped) {
    if (settings.bg_color_update_requested.exchange(false)) {
        glClearColor(Settings::values.bg_red.GetValue(), Settings::values.bg_green.GetValue(),
                     Settings::values.bg_blue.GetValue(), 0.0f);
    }
    if (settings.sampler_update_requested.exchange(false)) {
        ReloadSampler();
    }
    if (settings.shader_update_requested.exchange(false)) {
        shader.Release();
        ReloadShader();
    }

    const auto& top_screen = layout.top_screen;
    const auto& bottom_screen = layout.bottom_screen;

    glViewport(0, 0, layout.width, layout.height);
    glClear(GL_COLOR_BUFFER_BIT);

    // Set projection matrix
    std::array<GLfloat, 3 * 2> ortho_matrix =
        MakeOrthographicMatrix((float)layout.width, (float)layout.height, flipped);
    glUniformMatrix3x2fv(uniform_modelview_matrix, 1, GL_FALSE, ortho_matrix.data());

    // Bind texture in Texture Unit 0
    glUniform1i(uniform_color_texture, 0);

    const bool stereo_single_screen =
        Settings::values.render_3d.GetValue() == Settings::StereoRenderOption::Anaglyph ||
        Settings::values.render_3d.GetValue() == Settings::StereoRenderOption::Interlaced ||
        Settings::values.render_3d.GetValue() == Settings::StereoRenderOption::ReverseInterlaced;

    // Bind a second texture for the right eye if in Anaglyph mode
    if (stereo_single_screen) {
        glUniform1i(uniform_color_texture_r, 1);
    }

    glUniform1i(uniform_layer, 0);
    if (!Settings::values.swap_screen.GetValue()) {
        DrawTopScreen(layout, top_screen, stereo_single_screen);
        glUniform1i(uniform_layer, 0);
        ApplySecondLayerOpacity();
        DrawBottomScreen(layout, bottom_screen, stereo_single_screen);
    } else {
        DrawBottomScreen(layout, bottom_screen, stereo_single_screen);
        glUniform1i(uniform_layer, 0);
        ApplySecondLayerOpacity();
        DrawTopScreen(layout, top_screen, stereo_single_screen);
    }
    ResetSecondLayerOpacity();
}

void RendererOpenGL::ApplySecondLayerOpacity() {
    if (Settings::values.custom_layout &&
        Settings::values.custom_second_layer_opacity.GetValue() < 100) {
        state.blend.src_rgb_func = GL_CONSTANT_ALPHA;
        state.blend.src_a_func = GL_CONSTANT_ALPHA;
        state.blend.dst_a_func = GL_ONE_MINUS_CONSTANT_ALPHA;
        state.blend.dst_rgb_func = GL_ONE_MINUS_CONSTANT_ALPHA;
        state.blend.color.alpha = Settings::values.custom_second_layer_opacity.GetValue() / 100.0f;
    }
}

void RendererOpenGL::ResetSecondLayerOpacity() {
    if (Settings::values.custom_layout &&
        Settings::values.custom_second_layer_opacity.GetValue() < 100) {
        state.blend.src_rgb_func = GL_ONE;
        state.blend.dst_rgb_func = GL_ZERO;
        state.blend.src_a_func = GL_ONE;
        state.blend.dst_a_func = GL_ZERO;
        state.blend.color.alpha = 0.0f;
    }
}

void RendererOpenGL::DrawTopScreen(const Layout::FramebufferLayout& layout,
                                   const Common::Rectangle<u32>& top_screen,
                                   const bool stereo_single_screen) {
    if (!layout.top_screen_enabled) {
        return;
    }

    if (layout.is_rotated) {
        if (Settings::values.render_3d.GetValue() == Settings::StereoRenderOption::Off) {
            int eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
            DrawSingleScreenRotated(screen_infos[eye], (float)top_screen.left,
                                    (float)top_screen.top, (float)top_screen.GetWidth(),
                                    (float)top_screen.GetHeight());
        } else if (Settings::values.render_3d.GetValue() ==
                   Settings::StereoRenderOption::SideBySide) {
            DrawSingleScreenRotated(screen_infos[0], (float)top_screen.left / 2,
                                    (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                    (float)top_screen.GetHeight());
            glUniform1i(uniform_layer, 1);
            DrawSingleScreenRotated(screen_infos[1],
                                    ((float)top_screen.left / 2) + ((float)layout.width / 2),
                                    (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                    (float)top_screen.GetHeight());
        } else if (Settings::values.render_3d.GetValue() ==
                   Settings::StereoRenderOption::CardboardVR) {
            DrawSingleScreenRotated(screen_infos[0], layout.top_screen.left, layout.top_screen.top,
                                    layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
            glUniform1i(uniform_layer, 1);
            DrawSingleScreenRotated(
                screen_infos[1], layout.cardboard.top_screen_right_eye + ((float)layout.width / 2),
                layout.top_screen.top, layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
        } else if (stereo_single_screen) {
            DrawSingleScreenStereoRotated(screen_infos[0], screen_infos[1], (float)top_screen.left,
                                          (float)top_screen.top, (float)top_screen.GetWidth(),
                                          (float)top_screen.GetHeight());
        }
    } else {
        if (Settings::values.render_3d.GetValue() == Settings::StereoRenderOption::Off) {
            int eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
            DrawSingleScreen(screen_infos[eye], (float)top_screen.left, (float)top_screen.top,
                             (float)top_screen.GetWidth(), (float)top_screen.GetHeight());
        } else if (Settings::values.render_3d.GetValue() ==
                   Settings::StereoRenderOption::SideBySide) {
            DrawSingleScreen(screen_infos[0], (float)top_screen.left / 2, (float)top_screen.top,
                             (float)top_screen.GetWidth() / 2, (float)top_screen.GetHeight());
            glUniform1i(uniform_layer, 1);
            DrawSingleScreen(screen_infos[1],
                             ((float)top_screen.left / 2) + ((float)layout.width / 2),
                             (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                             (float)top_screen.GetHeight());
        } else if (Settings::values.render_3d.GetValue() ==
                   Settings::StereoRenderOption::CardboardVR) {
            DrawSingleScreen(screen_infos[0], layout.top_screen.left, layout.top_screen.top,
                             layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
            glUniform1i(uniform_layer, 1);
            DrawSingleScreen(
                screen_infos[1], layout.cardboard.top_screen_right_eye + ((float)layout.width / 2),
                layout.top_screen.top, layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
        } else if (stereo_single_screen) {
            DrawSingleScreenStereo(screen_infos[0], screen_infos[1], (float)top_screen.left,
                                   (float)top_screen.top, (float)top_screen.GetWidth(),
                                   (float)top_screen.GetHeight());
        }
    }
}

void RendererOpenGL::DrawBottomScreen(const Layout::FramebufferLayout& layout,
                                      const Common::Rectangle<u32>& bottom_screen,
                                      const bool stereo_single_screen) {
    if (!layout.bottom_screen_enabled) {
        return;
    }

    if (layout.is_rotated) {
        if (Settings::values.render_3d.GetValue() == Settings::StereoRenderOption::Off) {
            DrawSingleScreenRotated(screen_infos[2], (float)bottom_screen.left,
                                    (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                    (float)bottom_screen.GetHeight());
        } else if (Settings::values.render_3d.GetValue() ==
                   Settings::StereoRenderOption::SideBySide) {
            DrawSingleScreenRotated(screen_infos[2], (float)bottom_screen.left / 2,
                                    (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                    (float)bottom_screen.GetHeight());
            glUniform1i(uniform_layer, 1);
            DrawSingleScreenRotated(screen_infos[2],
                                    ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                                    (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                    (float)bottom_screen.GetHeight());
        } else if (Settings::values.render_3d.GetValue() ==
                   Settings::StereoRenderOption::CardboardVR) {
            DrawSingleScreenRotated(screen_infos[2], layout.bottom_screen.left,
                                    layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                    layout.bottom_screen.GetHeight());
            glUniform1i(uniform_layer, 1);
            DrawSingleScreenRotated(screen_infos[2],
                                    layout.cardboard.bottom_screen_right_eye +
                                        ((float)layout.width / 2),
                                    layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                    layout.bottom_screen.GetHeight());
        } else if (stereo_single_screen) {
            DrawSingleScreenStereoRotated(screen_infos[2], screen_infos[2],
                                          (float)bottom_screen.left, (float)bottom_screen.top,
                                          (float)bottom_screen.GetWidth(),
                                          (float)bottom_screen.GetHeight());
        }
    } else {
        if (Settings::values.render_3d.GetValue() == Settings::StereoRenderOption::Off) {
            DrawSingleScreen(screen_infos[2], (float)bottom_screen.left, (float)bottom_screen.top,
                             (float)bottom_screen.GetWidth(), (float)bottom_screen.GetHeight());
        } else if (Settings::values.render_3d.GetValue() ==
                   Settings::StereoRenderOption::SideBySide) {
            DrawSingleScreen(screen_infos[2], (float)bottom_screen.left / 2,
                             (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                             (float)bottom_screen.GetHeight());
            glUniform1i(uniform_layer, 1);
            DrawSingleScreen(screen_infos[2],
                             ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                             (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                             (float)bottom_screen.GetHeight());
        } else if (Settings::values.render_3d.GetValue() ==
                   Settings::StereoRenderOption::CardboardVR) {
            DrawSingleScreen(screen_infos[2], layout.bottom_screen.left, layout.bottom_screen.top,
                             layout.bottom_screen.GetWidth(), layout.bottom_screen.GetHeight());
            glUniform1i(uniform_layer, 1);
            DrawSingleScreen(screen_infos[2],
                             layout.cardboard.bottom_screen_right_eye + ((float)layout.width / 2),
                             layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                             layout.bottom_screen.GetHeight());
        } else if (stereo_single_screen) {
            DrawSingleScreenStereo(screen_infos[2], screen_infos[2], (float)bottom_screen.left,
                                   (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                   (float)bottom_screen.GetHeight());
        }
    }
}

void RendererOpenGL::TryPresent(int timeout_ms, bool is_secondary) {
    const auto& window = is_secondary ? *secondary_window : render_window;
    const auto& layout = window.GetFramebufferLayout();
    auto frame = window.mailbox->TryGetPresentFrame(timeout_ms);
    if (!frame) {
        LOG_DEBUG(Render_OpenGL, "TryGetPresentFrame returned no frame to present");
        return;
    }

    // Clearing before a full overwrite of a fbo can signal to drivers that they can avoid a
    // readback since we won't be doing any blending
    glClear(GL_COLOR_BUFFER_BIT);

    // Recreate the presentation FBO if the color attachment was changed
    if (frame->color_reloaded) {
        LOG_DEBUG(Render_OpenGL, "Reloading present frame");
        window.mailbox->ReloadPresentFrame(frame, layout.width, layout.height);
    }

    glWaitSync(frame->render_fence.handle, 0, GL_TIMEOUT_IGNORED);

    // INTEL workaround.
    // Normally we could just delete the draw fence here, but due to driver bugs, we can just delete
    // it on the emulation thread without too much penalty
    // glDeleteSync(frame.render_sync);
    // frame.render_sync = 0;

    glBindFramebuffer(GL_READ_FRAMEBUFFER, frame->present.handle);
    glBlitFramebuffer(0, 0, frame->width, frame->height, 0, 0, layout.width, layout.height,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);

    // Delete the fence if we're re-presenting to avoid leaking fences
    frame->present_fence.Release();

    // Insert fence for the main thread to block on
    frame->present_fence.Create();
    glFlush();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void RendererOpenGL::PrepareVideoDumping() {
    auto* mailbox = static_cast<OGLVideoDumpingMailbox*>(frame_dumper.mailbox.get());
    {
        std::unique_lock lock(mailbox->swap_chain_lock);
        mailbox->quit = false;
    }
    frame_dumper.StartDumping();
}

void RendererOpenGL::CleanupVideoDumping() {
    frame_dumper.StopDumping();
    auto* mailbox = static_cast<OGLVideoDumpingMailbox*>(frame_dumper.mailbox.get());
    {
        std::unique_lock lock(mailbox->swap_chain_lock);
        mailbox->quit = true;
    }
    mailbox->free_cv.notify_one();
}

void RendererOpenGL::Sync() {
    rasterizer.SyncEntireState();
}

} // namespace OpenGL
