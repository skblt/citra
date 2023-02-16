// Copyright 2019 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <QColorDialog>
#include "citra_qt/configuration/configuration_shared.h"
#include "citra_qt/configuration/configure_enhancements.h"
#include "common/settings.h"
#include "ui_configure_enhancements.h"
#include "video_core/renderer_opengl/post_processing_opengl.h"
#include "video_core/renderer_opengl/texture_filters/texture_filterer.h"

ConfigureEnhancements::ConfigureEnhancements(QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureEnhancements>()) {
    ui->setupUi(this);

    SetupPerGameUI();
    SetConfiguration();

    ui->layoutBox->setEnabled(!Settings::values.custom_layout);

    ui->resolution_factor_combobox->setEnabled(Settings::values.use_hw_renderer.GetValue());

    connect(ui->render_3d_combobox,
            static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
            [this](int currentIndex) {
                updateShaders(static_cast<Settings::StereoRenderOption>(currentIndex));
            });

    connect(ui->bg_button, &QPushButton::clicked, this, [this] {
        const QColor new_bg_color = QColorDialog::getColor(bg_color);
        if (!new_bg_color.isValid()) {
            return;
        }
        bg_color = new_bg_color;
        QPixmap pixmap(ui->bg_button->size());
        pixmap.fill(bg_color);
        const QIcon color_icon(pixmap);
        ui->bg_button->setIcon(color_icon);
    });

    ui->toggle_preload_textures->setEnabled(ui->toggle_custom_textures->isChecked());
    connect(ui->toggle_custom_textures, &QCheckBox::toggled, this, [this] {
        ui->toggle_preload_textures->setEnabled(ui->toggle_custom_textures->isChecked());
        if (!ui->toggle_preload_textures->isEnabled())
            ui->toggle_preload_textures->setChecked(false);
    });
}

ConfigureEnhancements::~ConfigureEnhancements() = default;

void ConfigureEnhancements::SetConfiguration() {

    if (!Settings::IsConfiguringGlobal()) {
        ConfigurationShared::SetHighlight(ui->widget_resolution,
                                          !Settings::values.resolution_factor.UsingGlobal());
        ConfigurationShared::SetHighlight(ui->widget_texture_filter,
                                          !Settings::values.texture_filter.UsingGlobal());
        ConfigurationShared::SetPerGameSetting(ui->resolution_factor_combobox,
                                               &Settings::values.resolution_factor);
        ConfigurationShared::SetPerGameSetting(ui->texture_filter_combobox,
                                               &Settings::values.texture_filter);
    } else {
        ui->resolution_factor_combobox->setCurrentIndex(
            Settings::values.resolution_factor.GetValue());
        ui->texture_filter_combobox->setCurrentIndex(
            static_cast<int>(Settings::values.texture_filter.GetValue()));
    }

    ui->render_3d_combobox->setCurrentIndex(
        static_cast<int>(Settings::values.render_3d.GetValue()));
    ui->factor_3d->setValue(Settings::values.factor_3d.GetValue());
    ui->mono_rendering_eye->setCurrentIndex(
        static_cast<int>(Settings::values.mono_render_option.GetValue()));
    updateShaders(Settings::values.render_3d.GetValue());
    ui->toggle_linear_filter->setChecked(Settings::values.linear_filter.GetValue());
    ui->layout_combobox->setCurrentIndex(
        static_cast<int>(Settings::values.layout_option.GetValue()));
    ui->swap_screen->setChecked(Settings::values.swap_screen.GetValue());
    ui->upright_screen->setChecked(Settings::values.upright_screen.GetValue());
    ui->toggle_dump_textures->setChecked(Settings::values.dump_textures.GetValue());
    ui->toggle_custom_textures->setChecked(Settings::values.custom_textures.GetValue());
    ui->toggle_preload_textures->setChecked(Settings::values.preload_textures.GetValue());
    bg_color =
        QColor::fromRgbF(Settings::values.bg_red.GetValue(), Settings::values.bg_green.GetValue(),
                         Settings::values.bg_blue.GetValue());
    QPixmap pixmap(ui->bg_button->size());
    pixmap.fill(bg_color);
    const QIcon color_icon(pixmap);
    ui->bg_button->setIcon(color_icon);
}

void ConfigureEnhancements::updateShaders(Settings::StereoRenderOption stereo_option) {
    ui->shader_combobox->clear();

    if (stereo_option == Settings::StereoRenderOption::Anaglyph)
        ui->shader_combobox->addItem(QStringLiteral("dubois (builtin)"));
    else if (stereo_option == Settings::StereoRenderOption::Interlaced ||
             stereo_option == Settings::StereoRenderOption::ReverseInterlaced)
        ui->shader_combobox->addItem(QStringLiteral("horizontal (builtin)"));
    else
        ui->shader_combobox->addItem(QStringLiteral("none (builtin)"));

    ui->shader_combobox->setCurrentIndex(0);

    for (const auto& shader : OpenGL::GetPostProcessingShaderList(
             stereo_option == Settings::StereoRenderOption::Anaglyph)) {
        ui->shader_combobox->addItem(QString::fromStdString(shader));
        if (Settings::values.pp_shader_name.GetValue() == shader)
            ui->shader_combobox->setCurrentIndex(ui->shader_combobox->count() - 1);
    }
}

void ConfigureEnhancements::RetranslateUI() {
    ui->retranslateUi(this);
}

void ConfigureEnhancements::ApplyConfiguration() {
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.resolution_factor,
                                             ui->resolution_factor_combobox);
    Settings::values.render_3d =
        static_cast<Settings::StereoRenderOption>(ui->render_3d_combobox->currentIndex());
    Settings::values.factor_3d = ui->factor_3d->value();
    Settings::values.mono_render_option =
        static_cast<Settings::MonoRenderOption>(ui->mono_rendering_eye->currentIndex());
    Settings::values.pp_shader_name =
        ui->shader_combobox->itemText(ui->shader_combobox->currentIndex()).toStdString();
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.linear_filter,
                                             ui->toggle_linear_filter, linear_filter);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.texture_filter,
                                             ui->texture_filter_combobox);
    Settings::values.layout_option =
        static_cast<Settings::LayoutOption>(ui->layout_combobox->currentIndex());
    Settings::values.swap_screen = ui->swap_screen->isChecked();
    Settings::values.upright_screen = ui->upright_screen->isChecked();

    ConfigurationShared::ApplyPerGameSetting(&Settings::values.dump_textures,
                                             ui->toggle_dump_textures, dump_textures);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.custom_textures,
                                             ui->toggle_custom_textures, custom_textures);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.preload_textures,
                                             ui->toggle_preload_textures, preload_textures);

    Settings::values.bg_red = static_cast<float>(bg_color.redF());
    Settings::values.bg_green = static_cast<float>(bg_color.greenF());
    Settings::values.bg_blue = static_cast<float>(bg_color.blueF());
}

void ConfigureEnhancements::SetupPerGameUI() {
    // Block the global settings if a game is currently running that overrides them
    if (Settings::IsConfiguringGlobal()) {
        ui->widget_resolution->setEnabled(Settings::values.resolution_factor.UsingGlobal());
        ui->widget_texture_filter->setEnabled(Settings::values.texture_filter.UsingGlobal());
        ui->toggle_linear_filter->setEnabled(Settings::values.linear_filter.UsingGlobal());
        ui->toggle_dump_textures->setEnabled(Settings::values.dump_textures.UsingGlobal());
        ui->toggle_custom_textures->setEnabled(Settings::values.custom_textures.UsingGlobal());
        ui->toggle_preload_textures->setEnabled(Settings::values.preload_textures.UsingGlobal());
        return;
    }

    ConfigurationShared::SetColoredTristate(ui->toggle_linear_filter,
                                            Settings::values.linear_filter, linear_filter);
    ConfigurationShared::SetColoredTristate(ui->toggle_dump_textures,
                                            Settings::values.dump_textures, dump_textures);
    ConfigurationShared::SetColoredTristate(ui->toggle_custom_textures,
                                            Settings::values.custom_textures, custom_textures);
    ConfigurationShared::SetColoredTristate(ui->toggle_preload_textures,
                                            Settings::values.preload_textures, preload_textures);

    ConfigurationShared::SetColoredComboBox(
        ui->resolution_factor_combobox, ui->widget_resolution,
        static_cast<u32>(Settings::values.resolution_factor.GetValue(true)));

    ConfigurationShared::SetColoredComboBox(
        ui->texture_filter_combobox, ui->widget_texture_filter,
        static_cast<u32>(Settings::values.texture_filter.GetValue(true)));
}
