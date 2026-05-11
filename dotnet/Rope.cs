/*
 * Rope.cs — C# binding for the ROPE framework.
 *
 * Wraps librope via P/Invoke for fast in-process interpolation queries, and
 * the rope CLI subprocess for forecast and server lifecycle management.
 *
 * Typical usage
 * -------------
 *     using RopeFramework;
 *
 *     var r = new Rope();
 *     r.Forecast("2024-02-09 00:00:00", horizon: 24);
 *
 *     using (r)  // opens handle, closes on dispose
 *     {
 *         var result = r.Get(
 *             time: new DateTime(2024, 2, 9, 6, 0, 0, DateTimeKind.Utc),
 *             lst: 7.5, lat: 45.0, altKm: 400.0);
 *         Console.WriteLine($"density={result.Density}  uncertainty={result.Uncertainty}");
 *     }
 */

using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("RopeTests")]

namespace RopeFramework;

public class RopeException : Exception
{
    public int Code { get; }

    private static readonly string[] s_names =
    [
        "ok", "no server", "no forecast cached", "time out of range",
        "spatial point out of range", "bad argument", "internal error",
    ];

    public RopeException(int code, string message)
        : base($"[{(code >= 0 && code < s_names.Length ? s_names[code] : code.ToString())}] {message}")
    {
        Code = code;
    }
}

public sealed class QueryResult
{
    public double Density     { get; init; }
    public double Uncertainty { get; init; }
}

public sealed class ForecastResult
{
    public string WindowStart { get; init; } = "";
    public string WindowEnd   { get; init; } = "";
}

public sealed unsafe class Rope : IDisposable
{
    public const int Hold   = 0;
    public const int Interp = 1;

    private const int ErrBufLen = 256;

    // -------------------------------------------------------------------------
    // Native delegate types
    // -------------------------------------------------------------------------

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate nint RopeOpenFn(byte* sockPath, byte* errBuf, int errLen);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int RopeQueryFn(
        nint interp, int mode,
        double timeUnix, double lst, double lat, double altKm,
        double* density, double* uncertainty,
        byte* errBuf, int errLen);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int RopeQueryBatchFn(
        nint interp, int mode, int n,
        double* timesUnix, double* lst, double* lat, double* altKm,
        double* density, double* uncertainty,
        byte* errBuf, int errLen);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void RopeCloseFn(nint interp);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int RopeServerStopFn(byte* sockPath, byte* errBuf, int errLen);

    // -------------------------------------------------------------------------
    // Fields
    // -------------------------------------------------------------------------

    private readonly nint              _lib;
    private readonly RopeOpenFn        _ropeOpen;
    private readonly RopeQueryFn       _ropeQuery;
    private readonly RopeQueryBatchFn  _ropeQueryBatch;
    private readonly RopeCloseFn       _ropeClose;
    private readonly RopeServerStopFn? _ropeServerStop;

    private readonly string? _socketPath;
    private readonly string? _exePath;
    private readonly string? _configPath;
    private nint _handle;
    private bool _disposed;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /// <param name="libPath">
    ///   Path to librope.so / librope.dylib / rope.dll.
    ///   Defaults to lib/librope.* relative to the package root.
    /// </param>
    /// <param name="exePath">
    ///   Path to the rope CLI executable.
    ///   Defaults to bin/rope relative to the package root.
    /// </param>
    /// <param name="socketPath">
    ///   Unix domain socket path. Null → platform default.
    /// </param>
    /// <param name="configPath">
    ///   Path to rope.conf. Defaults to config/rope.conf in the package root.
    /// </param>
    public Rope(
        string? libPath    = null,
        string? exePath    = null,
        string? socketPath = null,
        string? configPath = null)
    {
        // Rope.cs compiles to a DLL that lives in dotnet/ inside the package;
        // lib/ and bin/ are siblings at the package root.
        string root = PackageRoot();

        libPath ??= ResolveLibPath(root);
        exePath ??= ResolveExePath(root);

        _lib             = NativeLibrary.Load(libPath);
        _ropeOpen        = Load<RopeOpenFn>("rope_open");
        _ropeQuery       = Load<RopeQueryFn>("rope_query");
        _ropeQueryBatch  = Load<RopeQueryBatchFn>("rope_query_batch");
        _ropeClose       = Load<RopeCloseFn>("rope_close");
        _ropeServerStop  = TryLoad<RopeServerStopFn>("rope_server_stop");

        _socketPath = socketPath;
        _exePath    = exePath;
        _configPath = configPath;
    }

    private T Load<T>(string symbol) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(_lib, symbol));

    private T? TryLoad<T>(string symbol) where T : Delegate
    {
        if (NativeLibrary.TryGetExport(_lib, symbol, out nint ptr))
            return Marshal.GetDelegateForFunctionPointer<T>(ptr);
        return null;
    }

    // -------------------------------------------------------------------------
    // IDisposable
    // -------------------------------------------------------------------------

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Close();
        NativeLibrary.Free(_lib);
    }

    // -------------------------------------------------------------------------
    // Handle lifecycle
    // -------------------------------------------------------------------------

    /// <summary>Fetch the cached grid from the server and open an interpolation handle.</summary>
    public void Open()
    {
        if (_handle != 0) return;
        byte* err = stackalloc byte[ErrBufLen];

        nint handle;
        if (_socketPath is not null)
        {
            byte[] sock = Encoding.UTF8.GetBytes(_socketPath + '\0');
            fixed (byte* sockPtr = sock)
                handle = _ropeOpen(sockPtr, err, ErrBufLen);
        }
        else
        {
            handle = _ropeOpen(null, err, ErrBufLen);
        }

        if (handle == 0)
            throw new RopeException(1, ReadErr(err));
        _handle = handle;
    }

    /// <summary>Release the interpolation handle. Does not affect the server.</summary>
    public void Close()
    {
        if (_handle == 0) return;
        _ropeClose(_handle);
        _handle = 0;
    }

    /// <summary>Re-fetch the grid from the server (picks up a new forecast).</summary>
    public void Refresh()
    {
        Close();
        Open();
    }

    /// <summary>Send the exit command to the server, stopping it.</summary>
    public void Shutdown()
    {
        if (_ropeServerStop is null) return;
        byte* err = stackalloc byte[ErrBufLen];

        if (_socketPath is not null)
        {
            byte[] sock = Encoding.UTF8.GetBytes(_socketPath + '\0');
            fixed (byte* sockPtr = sock)
                _ropeServerStop(sockPtr, err, ErrBufLen);
        }
        else
        {
            _ropeServerStop(null, err, ErrBufLen);
        }
    }

    // -------------------------------------------------------------------------
    // Server commands (via CLI subprocess)
    // -------------------------------------------------------------------------

    /// <summary>Ask the server to run a forecast and cache the resulting grid.</summary>
    /// <param name="start">Forecast start time (UTC).</param>
    /// <param name="horizon">Forecast duration in hours.</param>
    public ForecastResult Forecast(DateTime start, int horizon) =>
        Forecast(start.ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss"), horizon);

    /// <summary>Ask the server to run a forecast and cache the resulting grid.</summary>
    /// <param name="start">ISO 8601 forecast start time string (UTC).</param>
    /// <param name="horizon">Forecast duration in hours.</param>
    public ForecastResult Forecast(string start, int horizon)
    {
        if (_exePath is null)
            throw new InvalidOperationException(
                "rope executable not found; pass exePath explicitly or check your package layout");

        var sb = new StringBuilder($"forecast --start \"{start}\" --horizon {horizon}");
        if (_configPath is not null)
            sb.Append($" --config \"{_configPath}\"");

        using var proc = Process.Start(new ProcessStartInfo(_exePath, sb.ToString())
        {
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            UseShellExecute        = false,
        })!;

        string stdout = proc.StandardOutput.ReadToEnd();
        string stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit();

        if (proc.ExitCode != 0)
            throw new RopeException(6, (stderr.Length > 0 ? stderr : stdout).Trim());

        // Take the last non-empty line — guards against preamble lines if a
        // stale server was replaced mid-flight.
        string? lastLine = null;
        foreach (var line in stdout.Split('\n'))
            if (line.Trim().Length > 0)
                lastLine = line.Trim();

        using var doc = JsonDocument.Parse(lastLine ?? "{}");
        var root = doc.RootElement;
        return new ForecastResult
        {
            WindowStart = root.TryGetProperty("window_start", out var ws) ? ws.GetString() ?? "" : "",
            WindowEnd   = root.TryGetProperty("window_end",   out var we) ? we.GetString() ?? "" : "",
        };
    }

    // -------------------------------------------------------------------------
    // Interpolation queries
    // -------------------------------------------------------------------------

    /// <summary>Query density and uncertainty at a single point.</summary>
    /// <param name="timeUnix">Query time as Unix timestamp (seconds since 1970-01-01T00:00:00 UTC).</param>
    /// <param name="lst">Local Solar Time, hours [0, 24).</param>
    /// <param name="lat">Geodetic latitude, degrees [-87.5, 87.5].</param>
    /// <param name="altKm">Geometric altitude, km [100, 980].</param>
    /// <param name="mode">Rope.Hold or Rope.Interp (default).</param>
    public QueryResult Get(double timeUnix, double lst, double lat, double altKm, int mode = Interp)
    {
        EnsureOpen();
        double density, uncertainty;
        byte* err = stackalloc byte[ErrBufLen];

        int rc = _ropeQuery(_handle, mode, timeUnix, lst, lat, altKm,
                            &density, &uncertainty, err, ErrBufLen);
        if (rc != 0) throw new RopeException(rc, ReadErr(err));

        return new QueryResult { Density = density, Uncertainty = uncertainty };
    }

    /// <summary>Query density and uncertainty at a single point.</summary>
    /// <param name="time">Query time (UTC).</param>
    /// <param name="lst">Local Solar Time, hours [0, 24).</param>
    /// <param name="lat">Geodetic latitude, degrees [-87.5, 87.5].</param>
    /// <param name="altKm">Geometric altitude, km [100, 980].</param>
    /// <param name="mode">Rope.Hold or Rope.Interp (default).</param>
    public QueryResult Get(DateTime time, double lst, double lat, double altKm, int mode = Interp) =>
        Get(ToUnix(time), lst, lat, altKm, mode);

    /// <summary>Query density and uncertainty at N points in one call.</summary>
    /// <param name="timesUnix">Array of N Unix timestamps.</param>
    /// <param name="lsts">Array of N Local Solar Time values.</param>
    /// <param name="lats">Array of N latitude values.</param>
    /// <param name="altsKm">Array of N altitude values.</param>
    /// <param name="mode">Rope.Hold or Rope.Interp (default), applied to all points.</param>
    public QueryResult[] GetBatch(
        double[] timesUnix, double[] lsts, double[] lats, double[] altsKm,
        int mode = Interp)
    {
        int n = timesUnix.Length;
        if (lsts.Length != n || lats.Length != n || altsKm.Length != n)
            throw new ArgumentException("all input arrays must have the same length");

        EnsureOpen();

        var density     = new double[n];
        var uncertainty = new double[n];
        byte* err = stackalloc byte[ErrBufLen];

        fixed (double* tPtr   = timesUnix,
                       lstPtr  = lsts,
                       latPtr  = lats,
                       altPtr  = altsKm,
                       denPtr  = density,
                       uncPtr  = uncertainty)
        {
            int rc = _ropeQueryBatch(_handle, mode, n,
                                     tPtr, lstPtr, latPtr, altPtr,
                                     denPtr, uncPtr, err, ErrBufLen);
            if (rc != 0) throw new RopeException(rc, ReadErr(err));
        }

        var results = new QueryResult[n];
        for (int i = 0; i < n; i++)
            results[i] = new QueryResult { Density = density[i], Uncertainty = uncertainty[i] };
        return results;
    }

    /// <summary>Query density and uncertainty at N points in one call.</summary>
    /// <param name="times">Array of N DateTime values (UTC).</param>
    /// <param name="lsts">Array of N Local Solar Time values.</param>
    /// <param name="lats">Array of N latitude values.</param>
    /// <param name="altsKm">Array of N altitude values.</param>
    /// <param name="mode">Rope.Hold or Rope.Interp (default), applied to all points.</param>
    public QueryResult[] GetBatch(
        DateTime[] times, double[] lsts, double[] lats, double[] altsKm,
        int mode = Interp)
    {
        var timesUnix = new double[times.Length];
        for (int i = 0; i < times.Length; i++)
            timesUnix[i] = ToUnix(times[i]);
        return GetBatch(timesUnix, lsts, lats, altsKm, mode);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private void EnsureOpen()
    {
        if (_handle == 0) Open();
    }

    private static string ReadErr(byte* buf)
    {
        int len = 0;
        while (len < ErrBufLen && buf[len] != 0) len++;
        return Encoding.UTF8.GetString(buf, len);
    }

    internal static double ToUnix(DateTime dt)
    {
        if (dt.Kind == DateTimeKind.Unspecified)
            dt = DateTime.SpecifyKind(dt, DateTimeKind.Utc);
        return (dt.ToUniversalTime() - DateTime.UnixEpoch).TotalSeconds;
    }

    private static string PackageRoot()
    {
        // The compiled DLL lives in dotnet/ inside the package; lib/ and bin/
        // are siblings at the package root — one level up.
        string? asmDir = Path.GetDirectoryName(typeof(Rope).Assembly.Location);
        return asmDir is not null ? Path.GetFullPath(Path.Combine(asmDir, "..")) : ".";
    }

    private static string ResolveLibPath(string root)
    {
        string[] candidates =
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? [Path.Combine(root, "bin", "rope.dll")]
            : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                ? [Path.Combine(root, "lib", "librope.dylib")]
                :
                [
                    Path.Combine(root, "lib",   "librope.so"),
                    Path.Combine(root, "build", "librope.so"),
                ];

        foreach (var c in candidates)
            if (File.Exists(c)) return c;

        throw new FileNotFoundException(
            "librope not found; pass libPath explicitly or check your package layout");
    }

    private static string? ResolveExePath(string root)
    {
        string[] candidates =
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? [Path.Combine(root, "bin", "rope.exe")]
                :
                [
                    Path.Combine(root, "bin",   "rope"),
                    Path.Combine(root, "build", "rope"),
                ];

        foreach (var c in candidates)
            if (File.Exists(c)) return c;

        return null;
    }
}
