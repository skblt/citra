# Definitions for all external bundled libraries

# Suppress warnings from external libraries
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W0)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMakeModules)
include(DownloadExternals)
include(ExternalProject)

# Boost
if (NOT USE_SYSTEM_BOOST)
    set(BOOST_ROOT "${CMAKE_SOURCE_DIR}/externals/boost")
    set(Boost_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/externals/boost")
    set(Boost_NO_SYSTEM_PATHS ON)
    add_library(boost INTERFACE)
    target_include_directories(boost SYSTEM INTERFACE ${Boost_INCLUDE_DIR})
endif()

# Boost::serialization
if (NOT USE_SYSTEM_BOOST)
    file(GLOB boost_serialization_SRC "${CMAKE_SOURCE_DIR}/externals/boost/libs/serialization/src/*.cpp")
    add_library(boost_serialization STATIC ${boost_serialization_SRC})
    target_link_libraries(boost_serialization PUBLIC boost)
endif()

# Add additional boost libs here; remember to ALIAS them in the root CMakeLists!

# Catch2
if (CITRA_TESTS AND NOT TARGET Catch2::Catch2)
    set(CATCH_INSTALL_DOCS OFF)
    set(CATCH_INSTALL_EXTRAS OFF)
    add_subdirectory(catch2 EXCLUDE_FROM_ALL)
endif()

# Crypto++
if (NOT TARGET cryptopp::cryptopp)
    set(CRYPTOPP_BUILD_DOCUMENTATION OFF)
    set(CRYPTOPP_BUILD_TESTING OFF)
    set(CRYPTOPP_INSTALL OFF)
    set(CRYPTOPP_SOURCES "${CMAKE_SOURCE_DIR}/externals/cryptopp")
    add_subdirectory(cryptopp-cmake)

    # HACK: The logic to set up the base include directory for CryptoPP does not work with Android SDK CMake 3.22.1.
    # Until there is a fixed version available, this code will detect and add in the proper include if it does not exist.
    if(ANDROID)
        message(STATUS "Applying CryptoPP include fix.")
        get_target_property(CRYPTOPP_INCLUDES cryptopp INTERFACE_INCLUDE_DIRECTORIES)
        foreach(CRYPTOPP_INCLUDE ${CRYPTOPP_INCLUDES})
            if("${CRYPTOPP_INCLUDE}" MATCHES "\\$<BUILD_INTERFACE:(.*)/cryptopp>")
                message(STATUS "Fixed include path: ${CMAKE_MATCH_1}")
                target_include_directories(cryptopp PUBLIC $<BUILD_INTERFACE:${CMAKE_MATCH_1}>)
                break()
            endif()
        endforeach()
    endif()
endif()

# fmt and Xbyak need to be added before dynarmic
# libfmt
if (NOT TARGET fmt::fmt)
    option(FMT_INSTALL "" ON)
    add_subdirectory(fmt)
endif()

# Xbyak
if ("x86_64" IN_LIST ARCHITECTURE AND NOT TARGET xbyak::xbyak)
    add_subdirectory(xbyak EXCLUDE_FROM_ALL)
endif()

# Dynarmic
if ("x86_64" IN_LIST ARCHITECTURE OR "arm64" IN_LIST ARCHITECTURE)
    set(DYNARMIC_TESTS OFF)
    set(DYNARMIC_FRONTENDS "A32")
    add_subdirectory(dynarmic EXCLUDE_FROM_ALL)
    add_library(dynarmic::dynarmic ALIAS dynarmic)
endif()

# getopt
if (MSVC)
    add_subdirectory(getopt)
endif()

# Glad
add_subdirectory(glad)

# inih
if (NOT TARGET inih::INIReader)
    add_subdirectory(inih)
endif()

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

if (NOT TARGET libusb::usb)
    add_subdirectory(libusb EXCLUDE_FROM_ALL)
endif()

# SDL2
if (ENABLE_SDL2 AND NOT CITRA_USE_SYSTEM_SDL2)
    add_subdirectory(sdl2)
endif()

# Zstandard
set(ZSTD_LEGACY_SUPPORT OFF)
set(ZSTD_BUILD_PROGRAMS OFF)
set(ZSTD_BUILD_SHARED OFF)
add_subdirectory(zstd/build/cmake EXCLUDE_FROM_ALL)
target_include_directories(libzstd_static INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/externals/zstd/lib>)

# ENet
if (NOT TARGET enet::enet)
    add_subdirectory(enet EXCLUDE_FROM_ALL)
    target_include_directories(enet INTERFACE ./enet/include)
    add_library(enet::enet ALIAS enet)
endif()

# Cubeb
if (ENABLE_CUBEB AND NOT TARGET cubeb::cubeb)
    set(BUILD_TESTS OFF CACHE BOOL "")
    add_subdirectory(cubeb EXCLUDE_FROM_ALL)
    add_library(cubeb::cubeb ALIAS cubeb)
endif()

# DiscordRPC
if (USE_DISCORD_PRESENCE AND NOT TARGET DiscordRPC::discord-rpc)
    add_subdirectory(discord-rpc EXCLUDE_FROM_ALL)
    target_include_directories(discord-rpc INTERFACE ./discord-rpc/include)
endif()

if (ENABLE_WEB_SERVICE AND NOT TARGET httplib::httplib)
    if (NOT WIN32)
        find_package(OpenSSL 1.1)
        if (OPENSSL_FOUND)
            set(OPENSSL_LIBRARIES OpenSSL::SSL OpenSSL::Crypto)
        endif()
    endif()

    if (WIN32 OR NOT OPENSSL_FOUND)
        # LibreSSL
        set(LIBRESSL_SKIP_INSTALL ON CACHE BOOL "")
        set(OPENSSLDIR "/etc/ssl/")
        add_subdirectory(libressl EXCLUDE_FROM_ALL)
        target_include_directories(ssl INTERFACE ./libressl/include)
        target_compile_definitions(ssl PRIVATE -DHAVE_INET_NTOP)
        get_directory_property(OPENSSL_LIBRARIES
            DIRECTORY libressl
            DEFINITION OPENSSL_LIBS)
    endif()

    # httplib
    add_library(httplib INTERFACE)
    target_include_directories(httplib INTERFACE ./httplib)
    target_compile_definitions(httplib INTERFACE -DCPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(httplib INTERFACE ${OPENSSL_LIBRARIES})
    if (WIN32)
        target_link_libraries(httplib INTERFACE crypt32 cryptui ws2_32)
    endif()
    add_library(httplib::httplib ALIAS httplib)
endif()

# JSON
if (ENABLE_WEB_SERVICE AND NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE ./json)
    add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
endif()

# cpp-jwt
if (ENABLE_WEB_SERVICE AND NOT TARGET cpp-jwt::cpp-jwt)
    add_library(cpp-jwt INTERFACE)
    target_include_directories(cpp-jwt INTERFACE ./cpp-jwt/include)
    target_compile_definitions(cpp-jwt INTERFACE CPP_JWT_USE_VENDORED_NLOHMANN_JSON)
    add_library(cpp-jwt::cpp-jwt ALIAS cpp-jwt)
endif()

if (ENABLE_WEB_SERVICE AND ANDROID)
    add_subdirectory(android-ifaddrs)
endif()

# lodepng
add_subdirectory(lodepng)

# (xperia64): Only use libyuv on Android b/c of build issues on Windows and mandatory JPEG
if(ANDROID)
    # libyuv
    add_subdirectory(libyuv)
    target_include_directories(yuv INTERFACE ./libyuv/include)
endif()
