// Copyright 2016 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <QColorDialog>
#ifdef __APPLE__
#include <QMessageBox>
#endif
#include "citra_qt/configuration/configure_graphics.h"
#include "core/core.h"
#include "core/settings.h"
#include "ui_configure_graphics.h"
#include "video_core/renderer_vulkan/vk_instance.h"

ConfigureGraphics::ConfigureGraphics(QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureGraphics>()) {
    ui->setupUi(this);
    DiscoverPhysicalDevices();
    SetConfiguration();

    const bool not_running = !Core::System::GetInstance().IsPoweredOn();
    const bool hw_renderer_enabled = ui->toggle_hw_renderer->isChecked();
    ui->toggle_hw_renderer->setEnabled(not_running);
    ui->hw_renderer_group->setEnabled(hw_renderer_enabled && not_running);
    ui->toggle_vsync_new->setEnabled(not_running);
    ui->graphics_api_combo->setEnabled(not_running);
    ui->toggle_shader_jit->setEnabled(not_running);
    ui->toggle_disk_shader_cache->setEnabled(hw_renderer_enabled && not_running);
    SetPhysicalDeviceComboVisibility(ui->graphics_api_combo->currentIndex());

    connect(ui->graphics_api_combo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureGraphics::SetPhysicalDeviceComboVisibility);

    connect(ui->toggle_hw_renderer, &QCheckBox::toggled, this, [this] {
        auto checked = ui->toggle_hw_renderer->isChecked();
        ui->hw_renderer_group->setEnabled(checked);
        ui->toggle_disk_shader_cache->setEnabled(checked && ui->toggle_hw_shader->isChecked());
    });

    ui->hw_shader_group->setEnabled(ui->toggle_hw_shader->isChecked());
    ui->toggle_disk_shader_cache->setEnabled(ui->toggle_hw_renderer->isChecked() &&
                                             ui->toggle_hw_shader->isChecked());

    connect(ui->toggle_hw_shader, &QCheckBox::toggled, this, [this] {
        const bool checked = ui->toggle_hw_shader->isChecked();
        ui->hw_shader_group->setEnabled(checked);
        ui->toggle_disk_shader_cache->setEnabled(checked);
    });

#ifdef __APPLE__
    connect(ui->toggle_hw_shader, &QCheckBox::stateChanged, this, [this](int state) {
        if (state == Qt::Checked) {
            ui->toggle_separable_shader->setEnabled(true);
        }
    });
    connect(ui->toggle_separable_shader, &QCheckBox::stateChanged, this, [this](int state) {
        if (state == Qt::Checked) {
            QMessageBox::warning(
                this, tr("Hardware Shader Warning"),
                tr("Separable Shader support is broken on macOS with Intel GPUs, and will cause "
                   "graphical issues "
                   "like showing a black screen.<br><br>The option is only there for "
                   "test/development purposes. If you experience graphical issues with Hardware "
                   "Shader, please turn it off."));
        }
    });
#else
    // TODO(B3N30): Hide this for macs with none Intel GPUs, too.
    ui->toggle_separable_shader->setVisible(false);
#endif
}

ConfigureGraphics::~ConfigureGraphics() = default;

void ConfigureGraphics::SetConfiguration() {
    ui->toggle_hw_renderer->setChecked(Settings::values.use_hw_renderer);
    ui->toggle_hw_shader->setChecked(Settings::values.use_hw_shader);
    ui->toggle_separable_shader->setChecked(Settings::values.separable_shader);
    ui->toggle_accurate_mul->setChecked(Settings::values.shaders_accurate_mul);
    ui->toggle_shader_jit->setChecked(Settings::values.use_shader_jit);
    ui->toggle_disk_shader_cache->setChecked(Settings::values.use_disk_shader_cache);
    ui->toggle_vsync_new->setChecked(Settings::values.use_vsync_new);
    ui->graphics_api_combo->setCurrentIndex(static_cast<int>(Settings::values.graphics_api));
    ui->physical_device_combo->setCurrentIndex(static_cast<int>(Settings::values.physical_device));
}

void ConfigureGraphics::ApplyConfiguration() {
    Settings::values.use_hw_renderer = ui->toggle_hw_renderer->isChecked();
    Settings::values.use_hw_shader = ui->toggle_hw_shader->isChecked();
    Settings::values.separable_shader = ui->toggle_separable_shader->isChecked();
    Settings::values.shaders_accurate_mul = ui->toggle_accurate_mul->isChecked();
    Settings::values.use_shader_jit = ui->toggle_shader_jit->isChecked();
    Settings::values.use_disk_shader_cache = ui->toggle_disk_shader_cache->isChecked();
    Settings::values.use_vsync_new = ui->toggle_vsync_new->isChecked();
    Settings::values.graphics_api =
        static_cast<Settings::GraphicsAPI>(ui->graphics_api_combo->currentIndex());
    Settings::values.physical_device = static_cast<u16>(ui->physical_device_combo->currentIndex());
}

void ConfigureGraphics::RetranslateUI() {
    ui->retranslateUi(this);
}

void ConfigureGraphics::DiscoverPhysicalDevices() {
    Vulkan::Instance instance{};
    const auto physical_devices = instance.GetPhysicalDevices();

    ui->physical_device_combo->clear();
    for (const vk::PhysicalDevice& physical_device : physical_devices) {
        const QString name = QString::fromLocal8Bit(physical_device.getProperties().deviceName);
        ui->physical_device_combo->addItem(name);
    }
}

void ConfigureGraphics::SetPhysicalDeviceComboVisibility(int index) {
    const auto graphics_api = static_cast<Settings::GraphicsAPI>(index);
    const bool is_visible = graphics_api == Settings::GraphicsAPI::Vulkan;
    ui->physical_device_label->setVisible(is_visible);
    ui->physical_device_combo->setVisible(is_visible);
}
