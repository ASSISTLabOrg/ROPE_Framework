#pragma once
// Visibility macro for shared-library symbols.
//
// In shared-library targets, define ROPE_BUILDING_SHARED during compilation
// of the library itself.  Consumers who link against the shared library
// should define ROPE_SHARED.  For static-library or header-only usage,
// neither is needed and ROPE_API expands to nothing.

#if defined(_WIN32)
#  if defined(ROPE_BUILDING_SHARED)
#    define ROPE_API __declspec(dllexport)
#  elif defined(ROPE_SHARED)
#    define ROPE_API __declspec(dllimport)
#  else
#    define ROPE_API
#  endif
#else
#  if defined(ROPE_BUILDING_SHARED)
#    define ROPE_API __attribute__((visibility("default")))
#  else
#    define ROPE_API
#  endif
#endif
