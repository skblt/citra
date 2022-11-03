// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#import <Cocoa/Cocoa.h>

#include "citra_qt/applesurfacehelper.h"

namespace AppleSurfaceHelper {

void* GetSurfaceLayer(void* surface) {
    NSView* view = static_cast<NSView*>(surface);
    return view.layer;
}

} // AppleSurfaceHelper
