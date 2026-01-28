#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Enforces atomic commits by ensuring each commit contains exactly one file.

.DESCRIPTION
    This script is used as a pre-push hook to ensure that each commit being
    pushed contains exactly one file. Multiple commits are allowed in a single
    push operation, but each commit must be atomic (one file per commit).

    This enforces RULE 11: A commit MUST have a maximum of 1 file.

.NOTES
    Author: Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>
    Version: 0.0.0
    Last Modified: 2026-01-28
    Platform: Windows only
    Requirements: pwsh 7.5.4

.EXAMPLE
    # Validates that all commits being pushed contain exactly one file each.
    .\restrict-pre-push-atomic.ps1

.EXIT CODES
    0 - Success (all commits are atomic or no commits to push)
    1 - Failure (one or more commits contain multiple or zero files)
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest

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

function Test-CommitFileCount {
    <#
    .SYNOPSIS
        Validates that all commits being pushed contain exactly one file.

    .DESCRIPTION
        This function checks if all commits in the push meet the atomic
        commit requirement by verifying that each commit contains exactly
        one file. Multiple commits are allowed, but each must be atomic.

    .OUTPUTS
        Boolean - Returns $true if all commits are valid, $false otherwise.

    .EXAMPLE
        if (Test-CommitFileCount) { Write-Host "All commits are atomic" }
        Validates the atomic commit requirement for all commits.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    try {
        Write-DebugLog -Scope "HOOK-PREPUSH" `
            -Message "Checking atomic commit requirements"

        $commitList = @(& git rev-list '@{upstream}..HEAD' 2>&1 | `
            Where-Object { $_ -and $_ -notmatch '^\s*$' })

        $commitCount = $commitList.Count

        Write-DebugLog -Scope "HOOK-PREPUSH" `
            -Message "Found $commitCount commits to push"

        if ($commitCount -eq 0) {
            Write-InfoLog -Scope "HOOK-PREPUSH" `
                -Message "No commits to push, validation skipped"

            return $true
        }

        $violationList = @()

        for ($commitIndex = 0; $commitIndex -lt $commitList.Count; `
            $commitIndex++) {
            $currentCommitHash = $commitList[$commitIndex]
            $commitShortHash = $currentCommitHash.Substring(0, 7)

            $changedFileList = @(& git diff-tree --no-commit-id `
                --name-only -r $currentCommitHash 2>&1 | `
                Where-Object { $_ -and $_ -notmatch '^\s*$' })
            $changedFileCount = $changedFileList.Count

            Write-DebugLog -Scope "COMMIT-VALIDATE" `
                -Message "Commit $commitShortHash has $changedFileCount files"

            if ($changedFileCount -ne 1) {
                $commitMessage = & git log -1 --pretty='%s' `
                    $currentCommitHash 2>&1 | Select-Object -First 1

                $violation = @{
                    CommitHash = $currentCommitHash
                    ShortHash = $commitShortHash
                    Message = $commitMessage
                    FileCount = $changedFileCount
                    Files = $changedFileList
                }

                $violationList += $violation
            }
        }

        if ($violationList.Count -gt 0) {
            $message = "ATOMIC_PUSH_FAILED: $($violationList.Count) " +
                "commit(s) violate the one-file-per-commit rule"
            Write-ErrorLog -Scope "HOOK-PREPUSH" -Message $message

            foreach ($violation in $violationList) {
                $violationMessage = "ATOMIC_COMMIT_VIOLATION: Commit " +
                    "$($violation.ShortHash) contains " +
                    "$($violation.FileCount) files (1 required)"
                Write-ErrorLog -Scope "HOOK-PREPUSH" `
                    -Message $violationMessage

                $messageText = "Message: $($violation.Message)"
                Write-ErrorLog -Scope "HOOK-PREPUSH" -Message $messageText

                if ($violation.FileCount -gt 0) {
                    Write-ErrorLog -Scope "HOOK-PREPUSH" -Message "Files:"
                    $violation.Files | ForEach-Object {
                        Write-ErrorLog -Scope "HOOK-PREPUSH" `
                            -Message "  - $_"
                    }
                }
            }

            return $false
        }

        Write-InfoLog -Scope "HOOK-PREPUSH" `
            -Message "All commits meet atomic requirements"

        return $true
    } catch {
        Write-ErrorLog -Scope "HOOK-PREPUSH" `
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
    $isValid = Test-CommitFileCount
    if (-not $isValid) {
        Write-ErrorLog -Scope "SCRIPT-MAIN" `
            -Message "Atomic commit validation failed"

        exit 1
    }

    Write-InfoLog -Scope "SCRIPT-MAIN" `
        -Message "Success: Pre-push validation completed"

    exit 0
} catch {
    Write-ExceptionLog -Scope "SCRIPT-MAIN" `
        -Message "Unexpected issue: $($_.Exception.Message)"

    Write-DebugLog -Scope "SCRIPT-MAIN" `
        -Message "Stack Trace: $($_.ScriptStackTrace)"

    exit 1
}

#endregion
