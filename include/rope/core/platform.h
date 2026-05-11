#pragma once
// Platform abstraction layer — OS-specific socket I/O and process management.
//
// Implementations live in src/core/platform/posix.cpp (Linux/macOS) and
// src/core/platform/windows.cpp (Windows).  No other source file may include
// OS-specific headers directly.

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <memory>

namespace rope::platform {

// ---------------------------------------------------------------------------
// IpcSocket — minimal reliable byte-stream connection over a Unix domain socket.
// ---------------------------------------------------------------------------
class IpcSocket {
public:
    virtual ~IpcSocket() = default;

    // Send exactly n bytes from buf.  Throws std::runtime_error on failure.
    virtual void send_all(const void* buf, std::size_t n) = 0;

    // Receive exactly n bytes into buf.  Throws std::runtime_error on failure.
    virtual void recv_all(void* buf, std::size_t n) = 0;

    // Connect to the Unix domain socket at path.  Throws on failure.
    static std::unique_ptr<IpcSocket> connect(const std::filesystem::path& path);
};

// ---------------------------------------------------------------------------
// ServerSocket — bind at a path, accept incoming connections.
// ---------------------------------------------------------------------------
class ServerSocket {
public:
    virtual ~ServerSocket() = default;

    // Block until a client connects or the running flag is cleared.
    // Returns the connected socket, or nullptr if running became false before a client arrived.
    virtual std::unique_ptr<IpcSocket> accept(const std::atomic<bool>& running) = 0;

    // Bind and listen at path.  Throws on failure.
    static std::unique_ptr<ServerSocket> bind(const std::filesystem::path& path);
};

// ---------------------------------------------------------------------------
// Per-user default socket path.
//   Linux/macOS: $XDG_RUNTIME_DIR/rope.sock, fallback /tmp/rope-<uid>.sock
//   Windows:     %LOCALAPPDATA%\rope\rope.sock
// ---------------------------------------------------------------------------
std::filesystem::path default_socket_path();

// ---------------------------------------------------------------------------
// Default directory for cached driver files written by DriverCacheManager.
//   Linux/macOS: $XDG_CACHE_HOME/rope/drivers or ~/.cache/rope/drivers
//   Windows:     %LOCALAPPDATA%\rope\drivers
// ---------------------------------------------------------------------------
std::filesystem::path default_cache_dir();

// ---------------------------------------------------------------------------
// Spawn `exe --serve --socket-path <socket_path> --config-path <config_path>`
// as a detached background process.  Returns before the process is ready.
// ---------------------------------------------------------------------------
void spawn_server(const std::filesystem::path& exe,
                  const std::filesystem::path& socket_path,
                  const std::filesystem::path& config_path);

} // namespace rope::platform
