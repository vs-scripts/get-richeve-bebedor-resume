<# :
@echo off
echo.
echo Error: This script must be run from a PowerShell version 7.5.4 terminal.
echo.
exit /b 1
#>

<#
.SYNOPSIS
    Configures Git to include the repository .gitconfig file locally.

.DESCRIPTION
    Sets the local Git include.path configuration so that the custom
    .gitconfig file in this repository is loaded for this repository only.

.NOTES
    Author: Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>
    Version: 0.0.0
    Last Modified: 2026-01-28
    Platform: Windows only
    Requirements: pwsh 7.5.4

.EXAMPLE
    .\set-custom-local-gitconfig.ps1
    Configures Git to include the repository .gitconfig file.

.EXIT CODES
    0 - Success
    1 - Failure (with error message)
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest

$scriptDirectoryPath = $PSScriptRoot
$conciseLogModulePath = Join-Path $scriptDirectoryPath 'concise-log.psm1'
$coreModulePath = Join-Path $scriptDirectoryPath 'powershell-core.psm1'

$conciseLogModulePath = [System.IO.Path]::GetFullPath($conciseLogModulePath)
$coreModulePath = [System.IO.Path]::GetFullPath($coreModulePath)

if (-not (Test-Path -LiteralPath $conciseLogModulePath)) {
    Write-Error 'Required module not found: concise-log.psm1'
    exit 1
}

if (-not (Test-Path -LiteralPath $coreModulePath)) {
    Write-Error 'Required module not found: powershell-core.psm1'
    exit 1
}

Import-Module -Name $conciseLogModulePath -Force -ErrorAction Stop
Import-Module -Name $coreModulePath -Force -ErrorAction Stop

#region Primary Functions

function Get-RepositoryRootPath {
    <#
    .SYNOPSIS
        Retrieves the root path of the repository.

    .DESCRIPTION
        Determines the repository root path based on the script's location.
        Since this script is located in the /scripts/ directory, the parent
        directory is considered the repository root.

    .NOTES
        This function relies on the physical location of the script file.

    .EXAMPLE
        $root = Get-RepositoryRootPath
        Returns "C:\Path\To\Repo"
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param()

    $currentDirectoryPath = Split-Path -Parent $PSScriptRoot
    return $currentDirectoryPath
}

function Get-CustomGitConfigPath {
    <#
    .SYNOPSIS
        Retrieves the absolute path to the custom .gitconfig file.

    .DESCRIPTION
        Constructs the full path to the .gitconfig file located at the
        repository root.

    .NOTES
        Depends on Get-RepositoryRootPath.

    .EXAMPLE
        $configPath = Get-CustomGitConfigPath
        Returns "C:\Path\To\Repo\.gitconfig"
    #>
    [CmdletBinding()]
    [OutputType([string])]
    param()

    $repositoryRootPath = Get-RepositoryRootPath
    $customGitConfigPath = Join-Path $repositoryRootPath '.gitconfig'

    return $customGitConfigPath
}

function Assert-CustomGitConfigExists {
    <#
    .SYNOPSIS
        Verifies that the custom .gitconfig file exists.

    .DESCRIPTION
        Checks for the existence of the .gitconfig file at the repository root.
        Throws an error if the file is missing.

    .NOTES
        This check prevents configuration errors from invalid paths.

    .EXAMPLE
        Assert-CustomGitConfigExists
        Throws error if .gitconfig is missing.
    #>
    [CmdletBinding()]
    param()

    $customGitConfigPath = Get-CustomGitConfigPath

    if (-not (Test-Path -LiteralPath $customGitConfigPath)) {
        Write-ErrorLog -Scope 'GIT-CONFIG' `
            -Message "Custom .gitconfig not found at $customGitConfigPath"

        throw "Custom .gitconfig not found at $customGitConfigPath"
    }
}

function Assert-GitCommandAvailable {
    <#
    .SYNOPSIS
        Verifies that the git command is available.

    .DESCRIPTION
        Checks if 'git' is in the system PATH and executable.
        Throws an error if git is not found.

    .NOTES
        Required for all git configuration operations.

    .EXAMPLE
        Assert-GitCommandAvailable
        Throws error if git is not installed.
    #>
    [CmdletBinding()]
    param()

    $gitCommand = Get-Command -Name 'git' -ErrorAction SilentlyContinue

    if (-not $gitCommand) {
        Write-ErrorLog -Scope 'GIT-CONFIG' `
            -Message "Required command not found: git"

        throw "Required command not found: git"
    }
}

function Set-CustomGitConfigIncludePath {
    <#
    .SYNOPSIS
        Sets the git local include.path to the repository .gitconfig.

    .DESCRIPTION
        Configures the local git environment to include the repository's
        custom configuration file. It is idempotent and checks if the
        configuration is already set before applying changes.

    .NOTES
        Modifies local git configuration.

    .EXAMPLE
        Set-CustomGitConfigIncludePath
        Updates git local config.
    #>
    [CmdletBinding()]
    param()

    $customGitConfigPath = '../.gitconfig'

    Write-InfoLog -Scope 'GIT-CONFIG' `
        -Message "Configuring Git include.path for $customGitConfigPath"

    $currentIncludePath = & git config --local include.path 2>$null

    if ($currentIncludePath) {
        Write-InfoLog -Scope 'GIT-CONFIG' `
            -Message "Removing existing include.path to ensure precedence"

        & git config --local --unset include.path
    }

    & git config --local include.path $customGitConfigPath

    Write-InfoLog -Scope 'GIT-CONFIG' `
        -Message "Git include.path set to $customGitConfigPath (at bottom)"
}

function Set-CustomLocalGitconfig {
    <#
    .SYNOPSIS
        Executes the main workflow for setting up git configuration.

    .DESCRIPTION
        Orchestrates the validation and configuration steps:
        1. Checks for custom .gitconfig existence.
        2. Checks for git command availability.
        3. Applies the global include.path configuration.

    .NOTES
        Main entry point for the script logic.

    .EXAMPLE
        Invoke-PrimaryWorkflow
        Runs the full configuration process.
    #>
    [CmdletBinding()]
    param()

    Assert-CustomGitConfigExists
    Assert-GitCommandAvailable
    Set-CustomGitConfigIncludePath

    Write-InfoLog -Scope 'GIT-CONFIG' `
        -Message "Success: Custom Git configuration is now active"
}

#endregion

#region Main Script Execution

Initialize-ScriptEnvironment
Assert-WindowsPlatform
Assert-PowerShellVersionStrict

try {
    Set-CustomLocalGitconfig

    exit 0
} catch {
    Write-ExceptionLog -Scope "GIT-CONFIG" `
        -Message "Unexpected issue: $($_.Exception.Message)"

    Write-DebugLog -Scope "GIT-CONFIG" `
        -Message "Stack Trace: $($_.ScriptStackTrace)"

    exit 1
}

#endregion
