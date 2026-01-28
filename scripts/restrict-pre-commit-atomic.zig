// restrict-pre-commit-atomic Command Dispatcher
//
// Purpose:
// This firmware-like component detects the operating system and executes
// the appropriate OS-specific script for restrict-pre-commit-atomic.
// It serves as a thin abstraction layer that forwards execution
// to platform-specific implementations.
//
// Contract:
// - No CLI argument parsing or interpretation
// - Must use safe and secured handling of memory allocation,
//   such as page allocation
// - No configuration handling
// - Single responsibility: OS detection and command execution
// - Process terminates after execution (no control regain)
//
// Platform Support & File Extension Mapping:
// This dispatcher explicitly associates OS platform types to specific file extensions
// to ensure type safety and predictable execution behavior:
//
// SUPPORTED PLATFORMS (by design - limited to 3 most popular):
// - Windows: Executes ONLY .ps1 files (PowerShell Core scripts)
//   - Requires PowerShell Core (pwsh) - NO backward compatibility with
//     Windows PowerShell 5.x (powershell.exe) due to complexity and
//     Microsoft's own deprecation strategy for legacy systems
//   - Rationale: Following Windows culture of dropping backward compatibility
//     for obsolete systems to reduce maintenance complexity
//
// - Linux: Executes ONLY .bash files (Bash shell scripts)
//   - Uses bash interpreter for consistent behavior across distributions
//   - Ensures compatibility with bash-specific features and syntax
//
// - macOS: Executes ONLY .sh files (POSIX shell scripts)
//   - Uses POSIX-compliant sh for maximum compatibility
//   - Avoids bash-specific features that may not be available
//
// UNSUPPORTED PLATFORMS (by design):
// Other Unix-like systems (FreeBSD, OpenBSD, etc.) are intentionally not supported
// to maintain simplicity and focus on the 3 most common/popular OS platforms.
// This is a conscious design decision to avoid complexity creep.
//
// Current Platform Stability:
// - Windows: STABLE - Full support with restrict-pre-commit-atomic.ps1
// - Linux: PLANNED - Platform supported, script implementation pending
// - macOS: PLANNED - Platform supported, script implementation pending
//
// Graceful Degradation:
// If a platform is supported but the corresponding script file does not exist,
// the dispatcher will log an informative message and exit gracefully with
// appropriate error codes, allowing developers to understand the current
// implementation status.
//
// Design Principles:
// - SOLID: Single Responsibility Principle
// - Firmware-like: Minimal, focused, deterministic
// - Minimal side effects: Deterministic execution flow with logging
// - Self-documenting: Clear purpose and behavior
// - Explicit platform contracts: Clear file extension to OS mapping
//
// Security Limitations (by design):
// - Command injection protection is LIMITED to basic path validation
// - If executable is placed in malicious directory, script paths could be exploited
// - TOCTOU (Time-of-Check-Time-of-Use) race condition exists between file validation and execution
//   * File could be replaced/modified between access() check and actual execution
//   * Mitigation is complex and out of scope for this simple dispatcher
//   * Risk is acceptable in controlled deployment environments with proper permissions
// - Users/developers must ensure secure deployment and directory permissions
// - Advanced security hardening is out of scope for this simple dispatcher

const std = @import("std");
const builtin = @import("builtin");

/// Platform-specific path separator
const PATH_SEPARATOR = if (builtin.os.tag == .windows) '\\' else '/';

/// Main entry point for the OS-specific command dispatcher
///
/// Execution flow:
/// 1. Extracts the command name from the executable filename
/// 2. Detects the current operating system
/// 3. Constructs the appropriate command for the detected OS
/// 4. Validates that the required script file exists for the platform
/// 5. Executes the command using execv (Unix) or std.process.Child (Windows)
/// 6. Terminates the process (execv replaces current process on Unix)
///
/// # Memory Management
/// All allocations use page_allocator with explicit defer statements for cleanup.
/// Allocations are bounded and process lifetime is short; OS reclaims all memory
/// on termination via exit() or execv().
///
/// NOTE: page_allocator is intentionally chosen over GeneralPurposeAllocator for this
/// short-lived dispatcher program because:
/// - Program has very short lifetime (seconds)
/// - Memory usage is minimal and bounded
/// - OS will reclaim all memory on process termination
/// - GeneralPurposeAllocator would add unnecessary complexity for debugging
/// This design decision is documented to prevent future "enhancement" attempts
/// that would add complexity without benefit for this specific use case.
///
/// # Error Handling & Exit Codes
/// The function uses specific exit codes for different failure scenarios:
/// - Exit Code 1: General execution failure (script execution, invalid arguments)
/// - Exit Code 2: Platform supported but script file not found
/// - Exit Code 3: Unsupported operating system
/// - Exit Code 4: System error (executable path, memory allocation)
///
/// # Possible Errors
/// Returns error if:
/// - ExecutablePathError: Cannot determine executable path
/// - InvalidScriptPath: Script path contains invalid characters
/// - UnsupportedOperatingSystem: Operating system not in supported list
/// - OutOfMemory: Memory allocation fails
/// - FileNotFound: Required script file does not exist for supported platform
/// - CommandExecutionError: Script execution fails
pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Retrieve the full path to this executable
    var exe_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const executable_path = std.fs.selfExePath(exe_path_buffer[0..]) catch {
        std.log.err("Failed to get executable path", .{});
        return error.ExecutablePathError;
    };

    // Use centralized path separator
    const sep = PATH_SEPARATOR;

    // Extract the filename portion from the full executable path
    // Example: "/path/to/restrict-pre-commit-atomic" -> "restrict-pre-commit-atomic"
    const filename_start_index = if (std.mem.lastIndexOfScalar(u8, executable_path, sep)) |index|
        index + 1
    else
        0; // If no separator found, use entire path as filename

    // Extract the command name without the file extension
    // Example: "restrict-pre-commit-atomic.exe" -> "restrict-pre-commit-atomic"
    // If no extension exists, use the entire filename
    const filename_slice = executable_path[filename_start_index..];
    const dot_index = std.mem.lastIndexOf(u8, filename_slice, ".") orelse filename_slice.len;
    const command_name = filename_slice[0..dot_index];

    // Capture all command-line arguments passed to this dispatcher
    const arguments = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, arguments);

    // Ensure we have at least one argument (argv[0])
    if (arguments.len == 0) {
        std.log.err("No arguments provided", .{});
        std.process.exit(4); // System error - invalid program state
    }

    // Build the full path to the OS-specific script
    const script_path = buildScriptPath(allocator, executable_path, command_name) catch |err| {
        switch (err) {
            error.UnsupportedOperatingSystem => {
                std.log.err("Unsupported operating system '{s}'. Supported platforms: Windows (.ps1), Linux (.bash), macOS (.sh)", .{@tagName(builtin.os.tag)});
                std.process.exit(3);
            },
            error.OutOfMemory => {
                std.log.err("Memory allocation failed while building script path for command '{s}'", .{command_name});
                std.process.exit(4);
            },
            error.InvalidScriptPath => {
                std.log.err("Invalid script path generated for command '{s}'", .{command_name});
                std.process.exit(4);
            },
        }
    };
    defer allocator.free(script_path);

    // Validate that the script file exists before attempting execution
    // This provides graceful degradation for supported platforms with pending implementations
    std.fs.cwd().access(script_path, .{}) catch |err| {
        switch (err) {
            error.FileNotFound => {
                const platform_name = switch (builtin.os.tag) {
                    .windows => "Windows",
                    .linux => "Linux",
                    .macos => "macOS",
                    else => "Unknown",
                };
                const extension = switch (builtin.os.tag) {
                    .windows => ".ps1",
                    .linux => ".bash",
                    .macos => ".sh",
                    else => "unknown",
                };
                std.log.err("Platform {s} is supported but script implementation is not yet available.", .{platform_name});
                std.log.err("Expected script file: {s}", .{script_path});
                std.log.err("This platform requires a {s} file for execution.", .{extension});
                std.log.err("Current implementation status: Windows (STABLE), Linux (PLANNED), macOS (PLANNED)", .{});
                std.process.exit(2);
            },
            else => {
                std.log.err("Cannot access script file '{s}': {}", .{ script_path, err });
                std.process.exit(4);
            },
        }
    };

    const command_prefix = determineOsSpecificCommand() catch |err| {
        std.log.err("Failed to determine command for operating system '{s}': {}", .{ @tagName(builtin.os.tag), err });
        std.process.exit(3); // Unsupported operating system
    };

    // Construct the complete argv array: [interpreter, args..., script_path, original_args...]
    // Size: command_prefix + script_path + remaining arguments (skip argv[0]) + null terminator
    const argv_len = command_prefix.len + 1 + (arguments.len - 1);
    var argv = try allocator.alloc(
        ?[]const u8,
        argv_len + 1, // +1 for null terminator required by execv
    );
    defer allocator.free(argv);

    for (0..command_prefix.len) |i| {
        argv[i] = command_prefix[i];
    }
    argv[command_prefix.len] = script_path;
    if (arguments.len > 1) {
        for (0..arguments.len - 1) |i| {
            argv[command_prefix.len + 1 + i] = arguments[1 + i];
        }
    }
    argv[argv_len] = null; // Null terminator required by execv

    // Execute the script with the appropriate method for the OS
    if (builtin.os.tag == .windows) {
        // Windows: Use std.process.Child to spawn and wait for the process
        // Convert nullable argv back to non-null for Windows (which doesn't need null terminator)
        var windows_argv = try allocator.alloc([]const u8, argv_len);
        defer allocator.free(windows_argv);
        for (0..argv_len) |i| {
            windows_argv[i] = argv[i].?; // Safe because we know these are non-null
        }

        var child = std.process.Child.init(windows_argv, allocator);
        const term = child.spawnAndWait() catch |err| {
            std.log.err("Failed to execute command '{s}': {}", .{ windows_argv[0], err });
            std.process.exit(1); // General execution failure
        };

        // Forward the exit code from the child process
        // Windows signal handling: Convert process termination to appropriate exit codes
        switch (term) {
            .Exited => |code| std.process.exit(code),
            .Signal => |sig| {
                // Standard Unix convention: 128 + signal number for signal termination
                // Capped at 255 (maximum exit code value) for signals > 127
                const exit_code = if (sig <= 127) @as(u8, @intCast(128 + sig)) else 255;
                std.log.err("Process terminated by signal {}: exit code {}", .{ sig, exit_code });
                std.process.exit(exit_code);
            },
            .Stopped => |sig| {
                // Process was stopped (suspended) by signal
                // Use same convention as signal termination
                const exit_code = if (sig <= 127) @as(u8, @intCast(128 + sig)) else 255;
                std.log.err("Process stopped by signal {}: exit code {}", .{ sig, exit_code });
                std.process.exit(exit_code);
            },
            .Unknown => {
                std.log.err("Process terminated with unknown status", .{});
                std.process.exit(1); // General execution failure
            },
        }
    } else {
        // Unix-like systems: Use execv to replace the current process
        // This is more efficient as it doesn't spawn a child process
        std.process.execv(allocator, argv) catch |err| {
            std.log.err("Failed to execute command '{s}': {}", .{ argv[0].?, err });
            std.process.exit(1); // General execution failure
        };
    }
    // Note: execv replaces the current process on Unix-like systems,
    // so execution never reaches this point on those platforms
}

/// Constructs the absolute path to the OS-specific script file
///
/// This function implements explicit platform-to-extension mapping to ensure
/// type safety and predictable execution behavior across different operating systems.
///
/// Platform-Extension Mapping:
/// - Windows → .ps1 (PowerShell Core scripts only)
/// - Linux → .bash (Bash shell scripts for distribution compatibility)
/// - macOS → .sh (POSIX shell scripts for maximum compatibility)
///
/// Path Construction Process:
/// 1. Extracts the directory containing the executable
/// 2. Determines the appropriate script extension for the OS
/// 3. Allocates and constructs the full script path
/// 4. Validates path safety (deferred: advanced path traversal protection)
///
/// Example paths constructed:
/// - Windows: "C:\path\to\restrict-pre-commit-atomic.ps1"
/// - Linux:   "/path/to/restrict-pre-commit-atomic.bash"
/// - macOS:   "/path/to/restrict-pre-commit-atomic.sh"
///
/// # Parameters
/// - allocator: Memory allocator for the path string (caller must free)
/// - executable_path: Full path to the current executable
/// - command_name: Name of the command without extension
///
/// # Returns
/// - Allocated string containing the full script path
/// - Caller is responsible for freeing the returned string
///
/// # Possible Errors
/// - UnsupportedOperatingSystem: Operating system not in supported platform list
/// - OutOfMemory: Memory allocation fails during path construction
/// - InvalidScriptPath: Path contains suspicious patterns (basic validation only)
///
/// # Security Note
/// Current implementation includes basic path validation. Advanced path traversal
/// protection (encoded variants, canonicalization) is deferred for future implementation.
fn buildScriptPath(
    allocator: std.mem.Allocator,
    executable_path: []const u8,
    command_name: []const u8,
) (std.mem.Allocator.Error || error{ UnsupportedOperatingSystem, InvalidScriptPath })![]u8 {
    // Use centralized path separator
    const sep = PATH_SEPARATOR;

    // Extract the directory portion of the executable path
    const exe_dir = if (std.mem.lastIndexOfScalar(u8, executable_path, sep)) |last_sep|
        executable_path[0..last_sep]
    else
        "."; // If no separator found, assume current directory

    // Determine the script extension based on explicit OS-to-extension mapping
    const extension = switch (builtin.os.tag) {
        .windows => ".ps1", // PowerShell Core scripts (pwsh) - no backward compatibility
        .linux => ".bash", // Bash shell scripts for consistent distribution support
        .macos => ".sh", // POSIX shell scripts for maximum macOS compatibility
        else => return error.UnsupportedOperatingSystem,
    };

    // Construct the script path using proper path joining
    const script_path = if (std.mem.eql(u8, exe_dir, "."))
        try std.fmt.allocPrint(allocator, "{s}{s}", .{ command_name, extension })
    else
        try std.fmt.allocPrint(allocator, "{s}{c}{s}{s}", .{ exe_dir, sep, command_name, extension });

    // Validate that the constructed path doesn't exceed system limits
    if (script_path.len >= std.fs.max_path_bytes) {
        allocator.free(script_path);
        std.log.err("Constructed script path exceeds system maximum ({} >= {}): {s}", .{ script_path.len, std.fs.max_path_bytes, script_path });
        return error.InvalidScriptPath;
    }

    // Basic validation: ensure the script path doesn't contain suspicious patterns
    // Note: This is basic protection only. Advanced path traversal protection
    // (handling encoded variants, canonicalization) is deferred for future implementation
    if (std.mem.indexOf(u8, script_path, "..") != null) {
        allocator.free(script_path);
        return error.InvalidScriptPath;
    }

    return script_path;
}

/// Determines the interpreter command and arguments for the current OS
///
/// Returns the appropriate shell interpreter and its configuration flags
/// needed to execute the OS-specific script. The script path is provided
/// separately by the caller and appended to this command.
///
/// This function implements the explicit platform-to-interpreter mapping
/// that corresponds to the file extension mapping in buildScriptPath().
///
/// Command structure by OS:
/// - Windows: ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
///   - Uses PowerShell Core (pwsh) exclusively
///   - No backward compatibility with Windows PowerShell 5.x (powershell.exe)
///   - Rationale: Follows Microsoft's deprecation strategy and reduces complexity
///   - Flags ensure reliable execution without user profile interference
///
/// - Linux: ["bash"]
///   - Uses bash interpreter for consistent behavior across distributions
///   - Ensures compatibility with bash-specific features and syntax
///   - Matches .bash file extension requirement
///
/// - macOS: ["sh"]
///   - Uses POSIX-compliant sh for maximum compatibility
///   - Avoids bash-specific features that may not be available
///   - Matches .sh file extension requirement
///
/// The script path is appended after these arguments by the caller.
///
/// # Returns
/// - Slice of strings containing the interpreter and its arguments
/// - Lifetime is static (compile-time constant)
///
/// # Possible Errors
/// - UnsupportedOperatingSystem: Operating system not in supported platform list
///
/// # Platform Support Status
/// - Windows: STABLE (full implementation available)
/// - Linux: PLANNED (interpreter ready, script implementation pending)
/// - macOS: PLANNED (interpreter ready, script implementation pending)
fn determineOsSpecificCommand() ![]const []const u8 {
    return switch (builtin.os.tag) {
        .windows => &[_][]const u8{
            "pwsh", // PowerShell Core (required - no backward compatibility with powershell.exe)
            "-NoProfile", // Skip profile loading for faster startup and reliability
            "-ExecutionPolicy", // Override execution policy for this invocation
            "Bypass", // Bypass execution policy restrictions
            "-File", // Execute the following file argument
        },
        .linux => &[_][]const u8{
            "bash", // Bash shell - matches .bash extension requirement
        },
        .macos => &[_][]const u8{
            "sh", // POSIX shell - matches .sh extension requirement
        },
        else => return error.UnsupportedOperatingSystem,
    };
}
