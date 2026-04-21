#pragma once
// Server module interface.
//
// Called by cli/main.cpp when the binary is invoked with --serve.
// Implementation lives in src/server/ and is not yet complete.

#include <filesystem>

namespace rope::server {

// Run the server: bind the socket at socket_path, load config from config_path,
// and serve requests until a clean exit is requested.
// Blocks until the server shuts down.
void run(const std::filesystem::path& socket_path,
         const std::filesystem::path& config_path);

} // namespace rope::server
