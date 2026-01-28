#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Enforces atomic commits by restricting staged files to one file.

.DESCRIPTION
    This script is used as a pre-commit hook to ensure that only a
    single file is staged for commit. This enforces the atomic commit
    convention where each commit should contain changes to exactly one
    file.

.NOTES
    Author: Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>
    Version: 0.0.0
    Last Modified: 2026-01-28
    Platform: Windows only
    Requirements: pwsh 7.5.4

.EXAMPLE
    # Validates that exactly one file is staged for commit.
    .\restrict-pre-commit-atomic.ps1

.EXIT CODES
    0 - Success (atomic commit allowed)
    1 - Failure (multiple files staged or hook error)
#>

[CmdletBinding()]
param()

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

function Test-AtomicCommit {
    <#
    .SYNOPSIS
        Checks if the current commit is atomic (one file only).

    .DESCRIPTION
        This function checks if the staged files meet the atomic
        commit requirement by verifying that exactly one file is
        staged for commit. Uses git to retrieve the list of staged
        files.

    .OUTPUTS
        System.Boolean. Returns $true if the commit is atomic, $false
        otherwise.

    .EXAMPLE
        if (Test-AtomicCommit) {
            Write-InfoLog -Scope "HOOK-PRECOMMIT" `
                -Message "Commit is atomic"
        }

    .NOTES
        This function validates the staged file count before commit.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    try {
        Write-DebugLog -Scope "HOOK-PRECOMMIT" `
            -Message "Checking staged files for atomic commit"

        $stagedFileOutput = & git diff --cached --name-only 2>&1
        $stagedFileList = @($stagedFileOutput | Where-Object {
            $_ -and $_ -notmatch '^\s*$'
        })
        $stagedFileCount = $stagedFileList.Count

        Write-DebugLog -Scope "HOOK-PRECOMMIT" `
            -Message "Staged file count: $stagedFileCount"

        if ($stagedFileCount -gt 1) {
            Write-ErrorLog -Scope "HOOK-PRECOMMIT" `
                -Message "$stagedFileCount files staged (1 max required)"

            foreach ($file in $stagedFileList) {
                Write-DebugLog -Scope "HOOK-PRECOMMIT" `
                    -Message "Staged file: $file"
            }

            return $false
        }

        Write-InfoLog -Scope "HOOK-PRECOMMIT" `
            -Message "Atomic commit validated"

        return $true
    }
    catch {
        Write-ErrorLog -Scope "HOOK-PRECOMMIT" `
            -Message "Validation error: $($_.Exception.Message)"

        throw
    }
}

#endregion

#region Main Script Execution

Initialize-ScriptEnvironment
Assert-WindowsPlatform
Assert-PowerShellVersionStrict

try {
    $isAtomicCommit = Test-AtomicCommit

    if (-not $isAtomicCommit) {
        Write-ErrorLog -Scope "SCRIPT-MAIN" `
            -Message "Commit rejected: multiple files staged"

        exit 1
    }

    Write-InfoLog -Scope "SCRIPT-MAIN" `
        -Message "Success: Atomic commit validated"

    exit 0
}
catch {
    Write-ExceptionLog -Scope "SCRIPT-MAIN" `
        -Message "Unexpected issue: $($_.Exception.Message)"

    Write-DebugLog -Scope "SCRIPT-MAIN" `
        -Message "Stack Trace: $($_.ScriptStackTrace)"

    exit 1
}

#endregion
