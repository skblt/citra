# Definitions for all external bundled libraries

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMakeModules)
include(DownloadExternals)
include(ExternalProject)

# Crypto++
set(CRYPTOPP_BUILD_TESTING OFF)
set(DISABLE_RDRAND TRUE)
add_subdirectory(cryptopp)

# Dynarmic
if (ARCHITECTURE_x86_64 OR ARCHITECTURE_ARM64)
    set(DYNARMIC_TESTS OFF)
    set(DYNARMIC_NO_BUNDLED_FMT OFF) # Until we upgrade dynarmic
    set(DYNARMIC_FRONTENDS "A32")
    add_subdirectory(dynarmic)
endif()

# getopt
if (MSVC)
    add_subdirectory(getopt)
endif()

# inih
add_subdirectory(inih)

# MicroProfile
add_library(microprofile INTERFACE)
target_include_directories(microprofile INTERFACE ./microprofile)

# Nihstro
add_library(nihstro-headers INTERFACE)
target_include_directories(nihstro-headers INTERFACE ./nihstro/include)

# Open Source Archives
add_subdirectory(open_source_archives)

# SoundTouch
add_subdirectory(soundtouch)
# The SoundTouch target doesn't export the necessary include paths as properties by default
target_include_directories(SoundTouch INTERFACE ./soundtouch/include)

# Teakra
add_subdirectory(teakra EXCLUDE_FROM_ALL)

# Cubeb
if (ENABLE_CUBEB)
    set(BUILD_TESTS OFF CACHE BOOL "")
    add_subdirectory(cubeb EXCLUDE_FROM_ALL)
endif()