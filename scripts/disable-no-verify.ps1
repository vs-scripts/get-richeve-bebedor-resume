<# :
@echo off
echo.
echo Error: This script must be run from a PowerShell terminal.
echo.
exit /b 1
#>

<#
.SYNOPSIS
    Prevents the use of --no-verify flag in git commands.

.DESCRIPTION
    This script checks command line arguments for the presence of the
    --no-verify flag and prevents its use by exiting with an error
    code. This enforces commit verification and prevents bypassing of
    pre-commit hooks.

.NOTES
    Author: Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>
    Version: 0.0.0
    Last Modified: 2026-01-28
    Platform: Windows only
    Requirements: pwsh 7.5.4

.EXAMPLE
    .\disable-no-verify.ps1 --no-verify
    Exits with error code 1 when --no-verify is detected.

.EXIT CODES
    0 - Success (no --no-verify flag found)
    1 - Failure (--no-verify flag detected or other error)
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string[]]$Arguments
)

# Import required modules
$scriptPath = $PSScriptRoot
$conciseLogPath = Join-Path $scriptPath 'concise-log.psm1'
$coreModulePath = Join-Path $scriptPath 'powershell-core.psm1'

# Convert to absolute paths (REQUIRED)
$conciseLogPath = [System.IO.Path]::GetFullPath($conciseLogPath)
$coreModulePath = [System.IO.Path]::GetFullPath($coreModulePath)

if (-not (Test-Path -LiteralPath $conciseLogPath)) {
    Write-Error 'Required module not found: concise-log.psm1'
    exit 1
}

if (-not (Test-Path -LiteralPath $coreModulePath)) {
    Write-Error 'Required module not found: powershell-core.psm1'
    exit 1
}

Import-Module -Name $conciseLogPath -Force -ErrorAction Stop
Import-Module -Name $coreModulePath -Force -ErrorAction Stop

#region Primary Functions

function Test-NoVerifyFlag {
    <#
    .SYNOPSIS
        Checks for --no-verify flag in arguments.

    .DESCRIPTION
        Analyzes command line arguments to detect and prevent the use
        of the --no-verify flag in git operations.

    .PARAMETER ArgumentList
        The command line arguments to check.

    .OUTPUTS
        System.Boolean. Returns $true if flag detected, $false
        otherwise.

    .EXAMPLE
        if (Test-NoVerifyFlag -ArgumentList $Arguments) {
            Write-ErrorLog -Scope "HOOK-VERIFY" `
                -Message "Flag --no-verify detected"
        }

    .NOTES
        This is the core functionality that enforces commit
        verification.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $false)]
        [string[]]$ArgumentList
    )

    try {
        Write-DebugLog -Scope "HOOK-VERIFY" `
            -Message "Checking arguments for --no-verify flag"

        if ($ArgumentList -contains '--no-verify') {
            Write-ErrorLog -Scope "HOOK-VERIFY" `
                -Message "Use of --no-verify is disabled"

            Write-DebugLog -Scope "HOOK-VERIFY" `
                -Message "Detected in arguments: $ArgumentList"

            return $true
        }

        Write-InfoLog -Scope "HOOK-VERIFY" `
            -Message "No --no-verify flag detected"

        return $false
    }
    catch {
        Write-ErrorLog -Scope "HOOK-VERIFY" `
            -Message "Error checking arguments: $($_.Exception.Message)"

        throw
    }
}

#endregion

#region Main Script Execution

Initialize-ScriptEnvironment
Assert-WindowsPlatform
Assert-PowerShellVersionStrict

try {
    $flagDetected = Test-NoVerifyFlag -ArgumentList $Arguments

    if ($flagDetected) {
        Write-InfoLog -Scope "SCRIPT-MAIN" `
            -Message "Operation blocked: --no-verify flag detected"

        exit 1
    }

    Write-InfoLog -Scope "SCRIPT-MAIN" `
        -Message "Success: Operation completed successfully"

    exit 0
}
catch {
    Write-ErrorLog -Scope "SCRIPT-MAIN" `
        -Message "Operation failed: $($_.Exception.Message)"

    Write-DebugLog -Scope "SCRIPT-MAIN" `
        -Message "Stack Trace: $($_.ScriptStackTrace)"

    exit 1
}

#endregion
