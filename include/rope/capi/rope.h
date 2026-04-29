/* include/rope/capi/rope.h — ROPE C API
 *
 * Thin shared-library interface for in-process density interpolation.
 * Call rope_open to fetch the cached forecast grid from the running server,
 * then query it locally via rope_query / rope_query_batch.
 *
 * This header must remain valid C.
 */

#ifndef ROPE_CAPI_H
#define ROPE_CAPI_H

#include "rope/core/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle — heap-allocated by rope_open, freed by rope_close. */
typedef struct rope_interp rope_interp_t;

/*
 * Connect to the server, fetch the cached grid, and return an interpolation handle.
 *
 * sock_path  — path to the server socket.  Pass NULL for the platform default.
 * err_buf    — buffer for a human-readable error message on failure.
 * err_len    — size of err_buf in bytes.
 *
 * Returns a valid handle on success, NULL on failure (err_buf is filled).
 */
ROPE_API rope_interp_t* rope_open(const char* sock_path,
                                   char*       err_buf,
                                   int         err_len);

/*
 * Query density and uncertainty at a single point.
 *
 * mode        — ROPE_HOLD or ROPE_INTERP.
 * time_unix   — query time as a Unix timestamp (seconds since 1970-01-01T00:00:00 UTC).
 * lst         — Local Solar Time, hours [0, 24).
 * lat         — geodetic latitude, degrees [-87.5, 87.5].
 * alt_km      — geometric altitude, km [100, 980].
 * density     — output: density in kg/m³.
 * uncertainty — output: uncertainty in kg/m³.
 * err_buf     — filled on failure.
 * err_len     — size of err_buf.
 *
 * Returns ROPE_OK on success, a non-zero error code on failure.
 */
ROPE_API int rope_query(rope_interp_t* interp,
                         int            mode,
                         double         time_unix,
                         double         lst,
                         double         lat,
                         double         alt_km,
                         double*        density,
                         double*        uncertainty,
                         char*          err_buf,
                         int            err_len);

/*
 * Query density and uncertainty at N points in one call.
 *
 * mode        — ROPE_HOLD or ROPE_INTERP, applied to all points.
 * n           — number of query points.
 * times_unix  — array of n Unix timestamps.
 * lst         — array of n LST values.
 * lat         — array of n latitude values.
 * alt_km      — array of n altitude values.
 * density     — caller-allocated output array of n density values.
 * uncertainty — caller-allocated output array of n uncertainty values.
 * err_buf     — filled on the first point that fails; remaining points skipped.
 * err_len     — size of err_buf.
 *
 * Returns ROPE_OK if all points succeeded; non-zero on the first failure.
 */
ROPE_API int rope_query_batch(rope_interp_t* interp,
                               int            mode,
                               int            n,
                               const double*  times_unix,
                               const double*  lst,
                               const double*  lat,
                               const double*  alt_km,
                               double*        density,
                               double*        uncertainty,
                               char*          err_buf,
                               int            err_len);

/*
 * Release the handle and all associated resources.
 * Does not communicate with the server.  Safe to call with NULL.
 */
ROPE_API void rope_close(rope_interp_t* interp);

/*
 * Send an exit command to the server, asking it to shut down.
 *
 * sock_path  — path to the server socket.  Pass NULL for the platform default.
 * err_buf    — buffer for a human-readable error message on failure.
 * err_len    — size of err_buf in bytes.
 *
 * Returns ROPE_OK on success, a non-zero error code on failure.
 */
ROPE_API int rope_server_stop(const char* sock_path,
                               char*       err_buf,
                               int         err_len);

/* Mode constants */
#define ROPE_HOLD   0
#define ROPE_INTERP 1

/* Return codes */
#define ROPE_OK                0
#define ROPE_ERR_NO_SERVER     1   /* could not connect to server socket */
#define ROPE_ERR_NO_FORECAST   2   /* server has no cached forecast */
#define ROPE_ERR_TIME_RANGE    3   /* query time outside forecast window */
#define ROPE_ERR_SPATIAL_RANGE 4   /* query point outside grid bounds */
#define ROPE_ERR_BAD_ARG       5   /* invalid argument (NULL pointer, etc.) */
#define ROPE_ERR_INTERNAL      6   /* unexpected internal failure */

#ifdef __cplusplus
}
#endif

#endif /* ROPE_CAPI_H */
