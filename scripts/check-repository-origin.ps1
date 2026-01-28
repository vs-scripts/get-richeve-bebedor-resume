<# :
@echo off
echo.
echo Error: This script must be run from a PowerShell terminal.
echo.
exit /b 1
#>

<#
.SYNOPSIS
    Validates the git remote URL against the ORIGIN file.

.DESCRIPTION
    Verifies that the current repository's git remote URL matches the
    expected URL defined in the ORIGIN file. This ensures the repository
    is configured to use the correct remote source. If the URLs do not
    match, the script exits with an error and provides corrective
    guidance.

.NOTES
    Author: Richeve Bebedor
    Version: 0.0.0
    Last Modified: 2026-01-28
    Platform: Windows only
    Requirements: pwsh 7.5.4

.EXAMPLE
    .\check-repository-origin.ps1
    Validates the git remote URL against the ORIGIN file.

.EXIT CODES
    0 - Success: Remote URL matches ORIGIN file
    1 - Failure: Remote URL does not match or validation error
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

function Get-RepositoryRoot {
    <#
    .SYNOPSIS
        Retrieves the root directory of the git repository.

    .DESCRIPTION
        Attempts to detect the git repository root using the git command.
        Falls back to the current working directory if git detection
        fails.

    .OUTPUTS
        [string] The repository root path.

    .EXAMPLE
        $root = Get-RepositoryRoot
        Returns the repository root path.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param()

    [string]$repositoryRoot = $PWD.Path

    $gitCommand = Get-Command -Name 'git' -ErrorAction SilentlyContinue
    if ($gitCommand) {
        try {
            $detectedRoot = (& git rev-parse --show-toplevel 2>$null)
            if ($detectedRoot -and (Test-Path -LiteralPath `
                    $detectedRoot)) {
                $message = "Detected Git repository root: $detectedRoot"
                Write-DebugLog -Scope "REPO-ORIGIN" -Message $message
                $repositoryRoot = $detectedRoot
            }
        } catch {
            $message = "Git root detection failed, using current directory"
            Write-DebugLog -Scope "REPO-ORIGIN" -Message $message
        }
    }

    return $repositoryRoot
}

function Get-AllowedRemoteUrl {
    <#
    .SYNOPSIS
        Reads the allowed remote URL from the ORIGIN file.

    .DESCRIPTION
        Reads the ORIGIN file from the repository root and returns the
        expected git remote URL. Throws an error if the file cannot be
        read or is empty.

    .PARAMETER RepositoryRoot
        The root directory of the repository.

    .OUTPUTS
        [string] The allowed remote URL from the ORIGIN file.

    .EXAMPLE
        $allowedUrl = Get-AllowedRemoteUrl `
            -RepositoryRoot "C:\Projects\MyRepo"
        Returns the allowed remote URL from the ORIGIN file.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true, `
            HelpMessage = "Repository root path")]
        [ValidateNotNullOrEmpty()]
        [string]$RepositoryRoot
    )

    try {
        $originFilePath = Join-Path -Path $RepositoryRoot `
            -ChildPath 'ORIGIN'
        $allowedRemoteUrl = Get-Content -LiteralPath $originFilePath `
            -ErrorAction Stop | ForEach-Object { $_.Trim() } | `
            Where-Object { $_ -ne '' } | Select-Object -First 1

        if (-not $allowedRemoteUrl) {
            throw "ORIGIN file is empty or contains no valid URL"
        }

        $message = "Allowed remote URL from ORIGIN: $allowedRemoteUrl"
        Write-DebugLog -Scope "REPO-ORIGIN" -Message $message
        return $allowedRemoteUrl
    } catch {
        $errorMsg = "Error reading ORIGIN file: " + `
            "$($_.Exception.Message)"
        throw $errorMsg
    }
}

function Get-CurrentRemoteUrl {
    <#
    .SYNOPSIS
        Retrieves the current git remote URL.

    .DESCRIPTION
        Executes the git command to get the current remote URL for the
        'origin' remote. Throws an error if the command fails.

    .OUTPUTS
        [string] The current git remote URL.

    .EXAMPLE
        $currentUrl = Get-CurrentRemoteUrl
        Returns the current git remote URL.
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param()

    try {
        $currentRemoteUrl = & git remote get-url origin 2>$null
        if (-not $currentRemoteUrl) {
            throw "No remote URL found for 'origin'"
        }

        $message = "Current git remote URL: $currentRemoteUrl"
        Write-DebugLog -Scope "REPO-ORIGIN" -Message $message
        return $currentRemoteUrl.Trim()
    } catch {
        $errorMsg = "Error retrieving git remote URL: " + `
            "$($_.Exception.Message)"
        throw $errorMsg
    }
}

function Test-RemoteUrlMatch {
    <#
    .SYNOPSIS
        Compares the current remote URL with the allowed URL.

    .DESCRIPTION
        Performs a string comparison between the current git remote URL
        and the expected URL from the ORIGIN file. Returns true if they
        match, false otherwise.

    .PARAMETER CurrentUrl
        The current git remote URL.

    .PARAMETER AllowedUrl
        The expected remote URL from the ORIGIN file.

    .OUTPUTS
        [bool] True if URLs match, false otherwise.

    .EXAMPLE
        $isMatch = Test-RemoteUrlMatch -CurrentUrl $current `
            -AllowedUrl $allowed
        Returns true if URLs match.
    #>
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $true, `
            HelpMessage = "Current remote URL")]
        [ValidateNotNullOrEmpty()]
        [string]$CurrentUrl,

        [Parameter(Mandatory = $true, `
            HelpMessage = "Allowed remote URL")]
        [ValidateNotNullOrEmpty()]
        [string]$AllowedUrl
    )

    return $CurrentUrl -eq $AllowedUrl
}

#endregion

#region Main Script Execution

Initialize-ScriptEnvironment
Assert-WindowsPlatform
Assert-PowerShellVersionStrict

try {
    Write-InfoLog -Scope "REPO-ORIGIN" `
        -Message "Validating remote URL"

    $repositoryRoot = Get-RepositoryRoot
    Write-DebugLog -Scope "REPO-ORIGIN" `
        -Message "Root: $repositoryRoot"

    $allowedRemoteUrl = Get-AllowedRemoteUrl `
        -RepositoryRoot $repositoryRoot
    $currentRemoteUrl = Get-CurrentRemoteUrl

    if (Test-RemoteUrlMatch -CurrentUrl $currentRemoteUrl `
            -AllowedUrl $allowedRemoteUrl) {
        Write-InfoLog -Scope "REPO-ORIGIN" `
            -Message "Remote URL valid"
        exit 0
    } else {
        Write-ErrorLog -Scope "REPO-ORIGIN" `
            -Message "Invalid remote URL"
        Write-WarningLog -Scope "REPO-ORIGIN" `
            -Message "Expected: $allowedRemoteUrl"
        Write-WarningLog -Scope "REPO-ORIGIN" `
            -Message "Found: $currentRemoteUrl"
        Write-InfoLog -Scope "REPO-ORIGIN" `
            -Message "Run: git remote set-url origin $allowedRemoteUrl"
        exit 1
    }
} catch {
    Write-ErrorLog -Scope "REPO-ORIGIN" -Message "Check failed"
    $errorMsg = "Error: $($_.Exception.Message)"
    Write-DebugLog -Scope "REPO-ORIGIN" -Message $errorMsg
    exit 1
}

#endregion
