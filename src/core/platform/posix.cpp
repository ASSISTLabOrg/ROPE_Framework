// POSIX platform implementation (Linux + macOS).
#include "rope/core/platform.h"

#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <spawn.h>
#include <fcntl.h>
#include <cerrno>
#include <climits>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef __APPLE__
#  include <mach-o/dyld.h>
#endif

extern char** environ;

namespace rope::platform {

// ---------------------------------------------------------------------------
// IpcSocket — POSIX file descriptor
// ---------------------------------------------------------------------------
class PosixSocket final : public IpcSocket {
public:
    explicit PosixSocket(int fd) : fd_(fd) {}
    ~PosixSocket() override { if (fd_ >= 0) ::close(fd_); }

    void send_all(const void* buf, std::size_t n) override {
        const auto* p = static_cast<const char*>(buf);
        while (n > 0) {
            ssize_t sent = ::send(fd_, p, n, MSG_NOSIGNAL);
            if (sent < 0)
                throw std::runtime_error(
                    std::string("IpcSocket::send_all: ") + std::strerror(errno));
            p += sent;
            n -= static_cast<std::size_t>(sent);
        }
    }

    void recv_all(void* buf, std::size_t n) override {
        auto* p = static_cast<char*>(buf);
        while (n > 0) {
            ssize_t got = ::recv(fd_, p, n, 0);
            if (got <= 0) {
                if (got == 0)
                    throw std::runtime_error("IpcSocket::recv_all: connection closed");
                throw std::runtime_error(
                    std::string("IpcSocket::recv_all: ") + std::strerror(errno));
            }
            p += got;
            n -= static_cast<std::size_t>(got);
        }
    }

private:
    int fd_;
};

// ---------------------------------------------------------------------------
// ServerSocket — POSIX bind/listen/accept
// ---------------------------------------------------------------------------
class PosixServerSocket final : public ServerSocket {
public:
    PosixServerSocket(int fd, std::filesystem::path path)
        : fd_(fd), path_(std::move(path)) {}

    ~PosixServerSocket() override {
        if (fd_ >= 0) {
            ::close(fd_);
            std::error_code ec;
            std::filesystem::remove(path_, ec);
        }
    }

    std::unique_ptr<IpcSocket> accept(const std::atomic<bool>& running) override {
        while (running) {
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(fd_, &rfds);
            timeval tv{1, 0};  // 1-second timeout to check stop flag
            int r = ::select(fd_ + 1, &rfds, nullptr, nullptr, &tv);
            if (r < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(
                    std::string("ServerSocket::accept: select: ") + std::strerror(errno));
            }
            if (r == 0) continue;  // timeout — loop to recheck running flag

            int client = ::accept(fd_, nullptr, nullptr);
            if (client < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(
                    std::string("ServerSocket::accept: accept: ") + std::strerror(errno));
            }
            return std::make_unique<PosixSocket>(client);
        }
        return nullptr;
    }

private:
    int                   fd_;
    std::filesystem::path path_;
};

std::unique_ptr<ServerSocket> ServerSocket::bind(const std::filesystem::path& path) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        throw std::runtime_error(
            std::string("ServerSocket::bind: socket(): ") + std::strerror(errno));

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    const std::string s = path.string();
    if (s.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        throw std::runtime_error("ServerSocket::bind: socket path too long");
    }
    std::strncpy(addr.sun_path, s.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(
            std::string("ServerSocket::bind: bind(): ") + std::strerror(e));
    }
    if (::listen(fd, 4) < 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(
            std::string("ServerSocket::bind: listen(): ") + std::strerror(e));
    }
    return std::make_unique<PosixServerSocket>(fd, path);
}

// ---------------------------------------------------------------------------
// IpcSocket::connect
// ---------------------------------------------------------------------------
std::unique_ptr<IpcSocket> IpcSocket::connect(const std::filesystem::path& path) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        throw std::runtime_error(
            std::string("IpcSocket::connect: socket(): ") + std::strerror(errno));

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    const std::string s = path.string();
    if (s.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        throw std::runtime_error("IpcSocket::connect: socket path too long");
    }
    std::strncpy(addr.sun_path, s.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(
            std::string("IpcSocket::connect: connect(): ") + std::strerror(e));
    }
    return std::make_unique<PosixSocket>(fd);
}

// ---------------------------------------------------------------------------
// exe_path
// ---------------------------------------------------------------------------
std::filesystem::path exe_path() {
#ifdef __linux__
    return std::filesystem::canonical("/proc/self/exe");
#else
    // macOS: _NSGetExecutablePath returns the real path of the running binary.
    char buf[PATH_MAX];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) {
        std::error_code ec;
        auto p = std::filesystem::canonical(buf, ec);
        return ec ? std::filesystem::path{buf} : p;
    }
    throw std::runtime_error("exe_path: _NSGetExecutablePath failed");
#endif
}

// ---------------------------------------------------------------------------
// default_cache_dir
// ---------------------------------------------------------------------------
std::filesystem::path default_cache_dir() {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"))
        return std::filesystem::path{xdg} / "rope" / "drivers";
    if (const char* home = std::getenv("HOME"))
        return std::filesystem::path{home} / ".cache" / "rope" / "drivers";
    return std::filesystem::temp_directory_path() / "rope" / "drivers";
}

// ---------------------------------------------------------------------------
// default_socket_path
// ---------------------------------------------------------------------------
std::filesystem::path default_socket_path() {
    if (const char* xdg = std::getenv("XDG_RUNTIME_DIR")) {
        return std::filesystem::path{xdg} / "rope.sock";
    }
    // Fallback: /tmp/rope-<uid>.sock
    std::string path = "/tmp/rope-" + std::to_string(::getuid()) + ".sock";
    return std::filesystem::path{path};
}

// ---------------------------------------------------------------------------
// spawn_server
// ---------------------------------------------------------------------------
void spawn_server(const std::filesystem::path& exe,
                  const std::filesystem::path& socket_path,
                  const std::filesystem::path& config_path) {
    std::string exe_s    = exe.string();
    std::string sock_s   = socket_path.string();
    std::string config_s = config_path.string();

    // Build null-terminated argv
    char* argv[] = {
        exe_s.data(),
        const_cast<char*>("--serve"),
        const_cast<char*>("--socket-path"),
        sock_s.data(),
        const_cast<char*>("--config-path"),
        config_s.data(),
        nullptr
    };

    // Redirect the server's stdin/stdout/stderr to /dev/null so it does not
    // inherit any captured pipes from the caller.  Without this,
    // subprocess.run(capture_output=True) in Python hangs: the server holds
    // the write end of the captured pipe open indefinitely.
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, STDIN_FILENO,  "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, "/dev/null", O_WRONLY, 0);

    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    // Request a new process group so the child survives the parent exiting
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);

    pid_t pid;
    int rc = posix_spawn(&pid, exe_s.c_str(), &fa, &attr, argv, environ);
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&fa);

    if (rc != 0)
        throw std::runtime_error(
            std::string("spawn_server: posix_spawn failed: ") + std::strerror(rc));
}

} // namespace rope::platform
