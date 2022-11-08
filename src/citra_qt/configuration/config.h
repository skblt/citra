// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <QVariant>
#include "common/settings.h"

class QSettings;

class Config {
public:
    Config();
    ~Config();

    void Reload();
    void Save();

    static const std::array<int, Settings::NativeButton::NumButtons> default_buttons;
    static const std::array<std::array<int, 5>, Settings::NativeAnalog::NumAnalogs> default_analogs;

private:
    void ReadValues();
    void ReadAudioValues();
    void ReadCameraValues();
    void ReadControlValues();
    void ReadCoreValues();
    void ReadDataStorageValues();
    void ReadDebuggingValues();
    void ReadLayoutValues();
    void ReadMiscellaneousValues();
    void ReadMultiplayerValues();
    void ReadPathValues();
    void ReadRendererValues();
    void ReadShortcutValues();
    void ReadSystemValues();
    void ReadUIValues();
    void ReadUIGameListValues();
    void ReadUILayoutValues();
    void ReadUpdaterValues();
    void ReadUtilityValues();
    void ReadWebServiceValues();
    void ReadVideoDumpingValues();

    void SaveValues();
    void SaveAudioValues();
    void SaveCameraValues();
    void SaveControlValues();
    void SaveCoreValues();
    void SaveDataStorageValues();
    void SaveDebuggingValues();
    void SaveLayoutValues();
    void SaveMiscellaneousValues();
    void SaveMultiplayerValues();
    void SavePathValues();
    void SaveRendererValues();
    void SaveShortcutValues();
    void SaveSystemValues();
    void SaveUIValues();
    void SaveUIGameListValues();
    void SaveUILayoutValues();
    void SaveUpdaterValues();
    void SaveUtilityValues();
    void SaveWebServiceValues();
    void SaveVideoDumpingValues();

    /**
     * Reads a setting from the qt_config.
     *
     * @param name The setting's identifier
     * @param default_value The value to use when the setting is not already present in the config
     */
    QVariant ReadSetting(const QString& name) const;
    QVariant ReadSetting(const QString& name, const QVariant& default_value) const;

    /**
     * Writes a setting to the qt_config.
     *
     * @param name The setting's idetentifier
     * @param value Value of the setting
     * @param default_value Default of the setting if not present in qt_config
     * @param use_global Specifies if the custom or global config should be in use, for custom
     * configs
     */
    void WriteSetting(const QString& name, const QVariant& value);
    void WriteSetting(const QString& name, const QVariant& value, const QVariant& default_value);

    std::unique_ptr<QSettings> qt_config;
    std::string qt_config_loc;
};
