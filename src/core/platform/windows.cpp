// Windows platform implementation.
#include "rope/core/platform.h"

// AF_UNIX sockets on Windows require Windows 10 1803+ and winsock2.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <afunix.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#pragma comment(lib, "ws2_32.lib")

namespace rope::platform {

// ---------------------------------------------------------------------------
// WSA initialisation — done once at process start
// ---------------------------------------------------------------------------
namespace {
struct WsaInit {
    WsaInit() {
        WSADATA wd{};
        if (WSAStartup(MAKEWORD(2, 2), &wd) != 0)
            throw std::runtime_error("IpcSocket: WSAStartup failed");
    }
    ~WsaInit() { WSACleanup(); }
} g_wsa;
} // namespace

// ---------------------------------------------------------------------------
// IpcSocket — Winsock SOCKET handle
// ---------------------------------------------------------------------------
class WinSocket final : public IpcSocket {
public:
    explicit WinSocket(SOCKET s) : s_(s) {}
    ~WinSocket() override { if (s_ != INVALID_SOCKET) closesocket(s_); }

    void send_all(const void* buf, std::size_t n) override {
        const auto* p = static_cast<const char*>(buf);
        while (n > 0) {
            int sent = ::send(s_, p, static_cast<int>(n), 0);
            if (sent == SOCKET_ERROR)
                throw std::runtime_error(
                    "IpcSocket::send_all: send error " +
                    std::to_string(WSAGetLastError()));
            p += sent;
            n -= static_cast<std::size_t>(sent);
        }
    }

    void recv_all(void* buf, std::size_t n) override {
        auto* p = static_cast<char*>(buf);
        while (n > 0) {
            int got = ::recv(s_, p, static_cast<int>(n), 0);
            if (got == 0)
                throw std::runtime_error("IpcSocket::recv_all: connection closed");
            if (got == SOCKET_ERROR)
                throw std::runtime_error(
                    "IpcSocket::recv_all: recv error " +
                    std::to_string(WSAGetLastError()));
            p += got;
            n -= static_cast<std::size_t>(got);
        }
    }

private:
    SOCKET s_;
};

// ---------------------------------------------------------------------------
// ServerSocket — Winsock bind/listen/accept
// ---------------------------------------------------------------------------
class WinServerSocket final : public ServerSocket {
public:
    WinServerSocket(SOCKET s, std::filesystem::path path)
        : s_(s), path_(std::move(path)) {}

    ~WinServerSocket() override {
        if (s_ != INVALID_SOCKET) {
            closesocket(s_);
            std::error_code ec;
            std::filesystem::remove(path_, ec);
        }
    }

    std::unique_ptr<IpcSocket> accept(const std::atomic<bool>& running) override {
        while (running) {
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(s_, &rfds);
            timeval tv{1, 0};
            // nfds is ignored by Winsock (pass 0 for clarity)
            int r = ::select(0, &rfds, nullptr, nullptr, &tv);
            if (r == SOCKET_ERROR) {
                int e = WSAGetLastError();
                if (e == WSAEINTR) continue;
                throw std::runtime_error(
                    "ServerSocket::accept: select error " + std::to_string(e));
            }
            if (r == 0) continue;

            SOCKET client = ::accept(s_, nullptr, nullptr);
            if (client == INVALID_SOCKET) {
                int e = WSAGetLastError();
                if (e == WSAEINTR) continue;
                throw std::runtime_error(
                    "ServerSocket::accept: accept error " + std::to_string(e));
            }
            return std::make_unique<WinSocket>(client);
        }
        return nullptr;
    }

private:
    SOCKET                s_;
    std::filesystem::path path_;
};

std::unique_ptr<ServerSocket> ServerSocket::bind(const std::filesystem::path& path) {
    // Ensure parent directory exists (needed for %LOCALAPPDATA%\rope\)
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    SOCKET s = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET)
        throw std::runtime_error(
            "ServerSocket::bind: socket error " + std::to_string(WSAGetLastError()));

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    const std::string ps = path.string();
    if (ps.size() >= sizeof(addr.sun_path)) {
        closesocket(s);
        throw std::runtime_error("ServerSocket::bind: socket path too long");
    }
    strncpy_s(addr.sun_path, sizeof(addr.sun_path), ps.c_str(), _TRUNCATE);

    if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        int e = WSAGetLastError();
        closesocket(s);
        throw std::runtime_error(
            "ServerSocket::bind: bind error " + std::to_string(e));
    }
    if (::listen(s, 4) == SOCKET_ERROR) {
        int e = WSAGetLastError();
        closesocket(s);
        throw std::runtime_error(
            "ServerSocket::bind: listen error " + std::to_string(e));
    }
    return std::make_unique<WinServerSocket>(s, path);
}

// ---------------------------------------------------------------------------
// IpcSocket::connect
// ---------------------------------------------------------------------------
std::unique_ptr<IpcSocket> IpcSocket::connect(const std::filesystem::path& path) {
    SOCKET s = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET)
        throw std::runtime_error(
            "IpcSocket::connect: socket() error " +
            std::to_string(WSAGetLastError()));

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    const std::string ps = path.string();
    if (ps.size() >= sizeof(addr.sun_path)) {
        closesocket(s);
        throw std::runtime_error("IpcSocket::connect: socket path too long");
    }
    strncpy_s(addr.sun_path, sizeof(addr.sun_path), ps.c_str(), _TRUNCATE);

    if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        int e = WSAGetLastError();
        closesocket(s);
        throw std::runtime_error(
            "IpcSocket::connect: connect() error " + std::to_string(e));
    }
    return std::make_unique<WinSocket>(s);
}

// ---------------------------------------------------------------------------
// default_socket_path
// ---------------------------------------------------------------------------
std::filesystem::path default_socket_path() {
    char buf[MAX_PATH];
    if (GetEnvironmentVariableA("LOCALAPPDATA", buf, MAX_PATH) == 0)
        throw std::runtime_error("default_socket_path: LOCALAPPDATA not set");
    return std::filesystem::path{buf} / "rope" / "rope.sock";
}

// ---------------------------------------------------------------------------
// spawn_server
// ---------------------------------------------------------------------------
void spawn_server(const std::filesystem::path& exe,
                  const std::filesystem::path& socket_path,
                  const std::filesystem::path& config_path) {
    std::string cmd =
        "\"" + exe.string() + "\" --serve"
        " --socket-path \"" + socket_path.string() + "\""
        " --config-path \"" + config_path.string() + "\"";

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    if (!CreateProcessA(
            nullptr,
            cmd.data(),
            nullptr, nullptr,
            FALSE,
            CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
            nullptr, nullptr,
            &si, &pi))
        throw std::runtime_error(
            "spawn_server: CreateProcess failed: " +
            std::to_string(GetLastError()));

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
}

} // namespace rope::platform
