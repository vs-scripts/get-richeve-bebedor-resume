<#
.SYNOPSIS
    Provides core functions for script initialization, platform validation,
    and elevation handling.

.DESCRIPTION
    The PowerShell Core Module provides essential functions for script
    initialization, platform validation, elevation handling, and formatted
    output. This module consolidates common functionality used across scripts,
    enabling code reuse and consistent behavior.

    The module includes:
    - Initialization functions
        - Initialize-ScriptEnvironment,
        - Assert-WindowsPlatform,
        - Test-IsInteractivePowerShell,
        - Invoke-PowerShellCoreTransition,
        - Assert-PowerShellVersionStrict
    - Elevation functions
        - Test-IsAdministrator,
        - Invoke-ElevationRequest

    Logging functionality is provided by the separate concise-log.psm1
    module.

.NOTES
    Author: PowerShell Core Module Team
    Version: 0.0.0
    Last Modified: 2026-01-26
    Platform: Windows only
    Requirements: pwsh 7.5.4
    Dependencies: concise-log.psm1

.EXAMPLE
    # Example: Non-elevated script structure
    $scriptPath = $PSScriptRoot
    $conciseLogPath = Join-Path $scriptPath 'concise-log.psm1'
    $coreModulePath = Join-Path $scriptPath 'powershell-core.psm1'

    # Convert to absolute paths (REQUIRED)
    $conciseLogPath = [System.IO.Path]::GetFullPath($conciseLogPath)
    $coreModulePath = [System.IO.Path]::GetFullPath($coreModulePath)

    Import-Module -Name $conciseLogPath -Force -ErrorAction Stop
    Import-Module -Name $coreModulePath -Force -ErrorAction Stop

    Initialize-ScriptEnvironment
    Assert-WindowsPlatform
    Invoke-PowerShellCoreTransition
    Assert-PowerShellVersionStrict

    # Example: Elevated script structure
    $scriptPath = $PSScriptRoot
    $conciseLogPath = Join-Path $scriptPath 'concise-log.psm1'
    $coreModulePath = Join-Path $scriptPath 'powershell-core.psm1'

    # Convert to absolute paths (REQUIRED)
    $conciseLogPath = [System.IO.Path]::GetFullPath($conciseLogPath)
    $coreModulePath = [System.IO.Path]::GetFullPath($coreModulePath)

    Import-Module -Name $conciseLogPath -Force -ErrorAction Stop
    Import-Module -Name $coreModulePath -Force -ErrorAction Stop

    Initialize-ScriptEnvironment
    Assert-WindowsPlatform
    Invoke-PowerShellCoreTransition
    Assert-PowerShellVersionStrict

    if (-not (Test-IsAdministrator)) {
        Invoke-ElevationRequest
    }

.EXIT CODES
    0 - Success
    1 - Failure (with error message)
#>

Set-StrictMode -Version Latest

# Import concise-log module for concise logging dependency
$modulePath = $PSScriptRoot
$conciseLogPath = Join-Path $modulePath 'concise-log.psm1'

# Convert to absolute path
$conciseLogPath = [System.IO.Path]::GetFullPath($conciseLogPath)

if (Test-Path -LiteralPath $conciseLogPath) {
    Import-Module -Name $conciseLogPath -Force -ErrorAction Stop -Global
} else {
    Write-Error "Required module not found: $conciseLogPath"
    throw "concise-log.psm1 module is required but not found"
}

#region Core Initialization Functions

function Initialize-ScriptEnvironment {
    <#
    .SYNOPSIS
        Initializes the script environment with standard PowerShell
        preferences.

    .DESCRIPTION
        Sets up PowerShell preferences for consistent behavior across all
        scripts. Configures StrictMode, ErrorActionPreference,
        DebugPreference, WarningPreference, and VerbosePreference to ensure
        predictable script execution.

    .OUTPUTS
        None. This function sets preferences and produces no output.

    .EXAMPLE
        Initialize-ScriptEnvironment
        Sets up PowerShell preferences for consistent behavior.

    .NOTES
        Context: Both elevated and non-elevated scripts
        This function should be called early in script execution to establish
        consistent preferences. It has no side effects beyond setting
        PowerShell preferences.
    #>
    [CmdletBinding()]
    param()

    try {
        Set-StrictMode -Version Latest
        $ErrorActionPreference = 'Stop'
        $DebugPreference = 'Continue'
        $WarningPreference = 'Continue'
        $VerbosePreference = 'Continue'
    }
    catch {
        Write-ErrorLog -Scope "INIT-ENV" `
            -Message "Failed to initialize script environment: $_"

        throw
    }
}

function Assert-WindowsPlatform {
    <#
    .SYNOPSIS
        Validates that the script is running on Windows platform.

    .DESCRIPTION
        Checks if the current PowerShell session is running on Windows
        platform. Throws an error if the script is not running on Windows,
        as many scripts require Windows-specific functionality.

    .OUTPUTS
        None. Throws an error if not on Windows platform.

    .EXAMPLE
        Assert-WindowsPlatform
        Validates that the script is running on Windows platform.

    .NOTES
        Context: Both elevated and non-elevated scripts
        This function should be called early in script execution to ensure
        the script is running on a compatible platform. It uses
        $PSVersionTable to determine the current platform.
    #>
    [CmdletBinding()]
    param()

    try {
        $isWindowsPlatform = $PSVersionTable.Platform -eq 'Win32NT' -or `
            $PSVersionTable.OS -like '*Windows*'

        if (-not $isWindowsPlatform) {
            throw "This script requires Windows platform. Current: " +
                "$($PSVersionTable.OS)"
        }
    }
    catch {
        Write-ErrorLog -Scope "ASSERT-PLATFORM" `
            -Message "Platform validation failed: $_"

        throw
    }
}


function Test-IsInteractivePowerShell {
    <#
    .SYNOPSIS
        Tests if the current PowerShell session is interactive.

    .DESCRIPTION
        Determines whether the current PowerShell session is interactive
        (user-driven) or non-interactive (automated, scheduled task, CI/CD
        pipeline). This is useful for functions that need different behavior
        in interactive vs. non-interactive contexts.

    .OUTPUTS
        System.Boolean. Returns $true if session is interactive, $false
        otherwise.

    .EXAMPLE
        if (Test-IsInteractivePowerShell) {
            Write-InfoLog -Scope "INTERACTIVE-TEST" `
                -Message "Running in interactive mode"
        } else {
            Write-InfoLog -Scope "INTERACTIVE-TEST" `
                -Message "Running in non-interactive mode"
        }

    .NOTES
        Context: Both elevated and non-elevated scripts
        This function checks $Environment.UserInteractive and input/output
        redirection to determine if the session is interactive.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    try {
        # Check if session is interactive
        $isInteractive = [Environment]::UserInteractive -and `
            -not [Environment]::GetCommandLineArgs().Contains(
                '-NonInteractive'
            )

        return $isInteractive
    }
    catch {
        Write-ErrorLog -Scope "TEST-INTERACTIVE" `
            -Message "Failed to determine if session is interactive: $_"

        return $false
    }
}

function Invoke-PowerShellCoreTransition {
    <#
    .SYNOPSIS
        Relaunches the script in PowerShell Core (pwsh) if needed.

    .DESCRIPTION
        Checks if the current PowerShell host is Windows PowerShell
        (powershell.exe) or if the pwsh version doesn't match 7.5.4. If
        transition is needed, relaunches the script in pwsh 7.5.4 with the
        same arguments and exits the current process.

    .OUTPUTS
        None. Exits the current process if relaunch is needed.

    .EXAMPLE
        Invoke-PowerShellCoreTransition
        Relaunches the script in PowerShell Core if needed.

    .NOTES
        Context: Both elevated and non-elevated scripts
        This function should be called early in script execution to ensure
        the script runs in the correct PowerShell version. It uses the call
        operator (&) to relaunch the script with the same arguments.
    #>
    [CmdletBinding()]
    param()

    try {
        # Check if running in Windows PowerShell
        $isWindowsPowerShell = $PSVersionTable.PSVersion.Major -lt 6

        if ($isWindowsPowerShell) {
            # Relaunch in pwsh
            $scriptPath = $MyInvocation.ScriptName
            $arguments = $PSBoundParameters.Values

            & pwsh -NoProfile -File $scriptPath @arguments
            exit $LASTEXITCODE
        }

        # Check if pwsh version matches 7.5.4
        $currentVersion = $PSVersionTable.PSVersion
        $requiredVersion = [version]'7.5.4'

        if ($currentVersion -ne $requiredVersion) {
            Write-WarningLog -Scope "PWSH-TRANSITION" `
                -Message "PowerShell version mismatch. Current: " +
                "$currentVersion, Required: $requiredVersion"
        }
    }
    catch {
        Write-ErrorLog -Scope "PWSH-TRANSITION" `
            -Message "Failed to transition to PowerShell Core: $_"

        throw
    }
}

function Assert-PowerShellVersionStrict {
    <#
    .SYNOPSIS
        Enforces exact PowerShell version 7.5.4.

    .DESCRIPTION
        Validates that the current PowerShell version is exactly 7.5.4.
        Throws an error if the version doesn't match exactly. This is
        stricter than Invoke-PowerShellCoreTransition and is used when exact
        version matching is required.

    .OUTPUTS
        None. Throws an error if version doesn't match.

    .EXAMPLE
        Assert-PowerShellVersionStrict
        Enforces exact PowerShell version 7.5.4.

    .NOTES
        Context: Both elevated and non-elevated scripts
        This function should be called when exact version matching is
        required. It checks $PSVersionTable.PSVersion against the required
        version 7.5.4.
    #>
    [CmdletBinding()]
    param()

    try {
        $currentVersion = $PSVersionTable.PSVersion
        $requiredVersion = [version]'7.5.4'

        if ($currentVersion -ne $requiredVersion) {
            throw "PowerShell version mismatch. Current: " +
                "$currentVersion, Required: $requiredVersion"
        }
    }
    catch {
        Write-ErrorLog -Scope "ASSERT-VERSION" `
            -Message "PowerShell version validation failed: $_"

        throw
    }
}

#endregion

#region Elevation Functions

function Test-IsAdministrator {
    <#
    .SYNOPSIS
        Tests if the current process has administrative privileges.

    .DESCRIPTION
        Determines whether the current PowerShell process is running with
        administrative privileges. This is useful for scripts that require
        elevation to perform their tasks.

    .OUTPUTS
        System.Boolean. Returns $true if running as administrator, $false
        otherwise.

    .EXAMPLE
        if (Test-IsAdministrator) {
            Write-Host "Running with administrative privileges"
        } else {
            Write-Host "Not running with administrative privileges"
        }

    .NOTES
        Context: Elevated scripts only
        This function uses Windows identity checks to determine
        administrative status. It works on Windows platforms only.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    try {
        $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal(
            $currentUser
        )
        $isAdmin = $principal.IsInRole(
            [Security.Principal.WindowsBuiltInRole]::Administrator
        )
        return $isAdmin
    }
    catch {
        Write-ErrorLog -Scope "TEST-ADMIN" `
            -Message "Failed to determine administrator status: $_"

        return $false
    }
}

function Invoke-ElevationRequest {
    <#
    .SYNOPSIS
        Requests elevation via UAC and relaunches the script.

    .DESCRIPTION
        Requests administrative privileges via User Account Control (UAC)
        and relaunches the script with elevated privileges. In non-interactive
        contexts (scheduled tasks, CI/CD pipelines), logs guidance and exits
        with code 1 instead of attempting elevation.

    .PARAMETER ScriptPath
        The full path to the script to relaunch with elevation.
        If not provided, attempts to determine from call stack.

    .OUTPUTS
        None. Exits the current process after relaunch or with code 1 in
        non-interactive context.

    .EXAMPLE
        if (-not (Test-IsAdministrator)) {
            Invoke-ElevationRequest -ScriptPath $PSCommandPath
        }

    .NOTES
        Context: Elevated scripts only
        This function checks if the session is interactive before attempting
        elevation. In non-interactive contexts, it logs guidance and exits
        with code 1.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $false)]
        [string]$ScriptPath
    )

    try {
        $isInteractive = Test-IsInteractivePowerShell

        if (-not $isInteractive) {
            Write-InfoLog -Scope "ELEV-REQUEST" `
                -Message "Cannot request elevation in non-interactive context"

            exit 1
        }

        # Determine script path if not provided
        if (-not $ScriptPath) {
            $callerInfo = Get-PSCallStack | Select-Object -Skip 1 -First 1
            $ScriptPath = $callerInfo.ScriptName
        }

        if (-not $ScriptPath -or -not (Test-Path -LiteralPath $ScriptPath)) {
            Write-ErrorLog -Scope "ELEV-REQUEST" `
                -Message "Cannot determine script path for elevation"
            exit 1
        }

        Write-InfoLog -Scope "ELEV-REQUEST" `
            -Message "Requesting administrative privileges"

        # Check for pwsh command
        $pwshCommand = Get-Command -Name 'pwsh' -ErrorAction SilentlyContinue

        if (-not $pwshCommand) {
            Write-ErrorLog -Scope "ELEV-REQUEST" `
                -Message "Elevation failed: 'pwsh' command not found in PATH"
            exit 1
        }

        # Relaunch with elevation using pwsh
        $argumentList = "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""

        $processInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processInfo.FileName = "pwsh"
        $processInfo.Arguments = $argumentList
        $processInfo.UseShellExecute = $true
        $processInfo.Verb = "runas"

        $process = [System.Diagnostics.Process]::Start($processInfo)
        $process.WaitForExit()

        exit $process.ExitCode
    }
    catch {
        Write-ErrorLog -Scope "ELEV-REQUEST" `
            -Message "Failed to request elevation: $_"

        exit 1
    }
}

#endregion

# Export public functions (7 functions)
Export-ModuleMember -Function @(
    'Initialize-ScriptEnvironment'
    'Assert-WindowsPlatform'
    'Test-IsInteractivePowerShell'
    'Invoke-PowerShellCoreTransition'
    'Assert-PowerShellVersionStrict'
    'Test-IsAdministrator'
    'Invoke-ElevationRequest'
)
