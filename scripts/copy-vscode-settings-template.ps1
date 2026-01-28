<# :
@echo off
echo.
echo Error: This script must be run from a PowerShell terminal.
echo.
exit /b 1
#>

<#
.SYNOPSIS
    Copies the VS Code settings template to the .vscode directory.

.DESCRIPTION
    This script copies the settings.json.template file from the template directory
    to the .vscode/settings.json location, enabling consistent VS Code configuration
    across the development environment.

.NOTES
    Author: Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>
    Version: 0.0.0
    Platform: Windows only
    Requirements: PowerShell 5.1 or later (pwsh 7+ preferred)

.EXAMPLE
    .\copy-vscode-settings-template.ps1
    Copies the template to .vscode/settings.json in the repository root.

.EXIT CODES
    0 - Success
    1 - Failure (with error message)
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest

# --- Core Functions ---

function Initialize-ScriptEnvironment {
    <#
    .SYNOPSIS
        Configures the PowerShell session preferences for consistent script behavior.

    .DESCRIPTION
        Sets script-level preferences for Verbose, Debug, ErrorAction, and Progress
        to ensure consistent and informative output throughout script execution.
        These settings apply only to the current script scope.

    .NOTES
        This function must be called early in script execution, before any other
        operations that depend on these preferences.

    .EXAMPLE
        Initialize-ScriptEnvironment
        Configures all session preferences to their standard values.
    #>
    [CmdletBinding()]
    param()

    $script:VerbosePreference = 'Continue'
    $script:DebugPreference = 'Continue'
    $script:ErrorActionPreference = 'Stop'
    $script:ProgressPreference = 'SilentlyContinue'
}

function Assert-WindowsPlatform {
    <#
    .SYNOPSIS
        Ensures the script is running on a Windows platform.

    .DESCRIPTION
        Checks the $PSVersionTable.Platform or $env:OS environment variable.
        Throws an exception if the platform is not Windows.

    .NOTES
        VS Code configuration in this script is currently targeted at Windows users.

    .EXAMPLE
        Assert-WindowsPlatform
        Validates that the current platform is Windows.
    #>
    [CmdletBinding()]
    param()

    $isWindowsPlatform = ($PSVersionTable.Platform -eq 'Win32NT') -or
        ($env:OS -eq 'Windows_NT')
    if (-not $isWindowsPlatform) {
        throw "This script is currently Windows-only."
    }
}

function Test-IsInteractivePowerShell {
    <#
    .SYNOPSIS
        Verifies if the script is running in an interactive PowerShell host.

    .DESCRIPTION
        Checks the $Host name to ensure the script isn't running in a
        non-interactive background process where user interaction might fail.

    .EXAMPLE
        Test-IsInteractivePowerShell
        Validates that the current session is interactive.
    #>
    [CmdletBinding()]
    param()

    if ($null -eq $Host -or $Host.Name -eq "Default Host") {
        Write-Error -Message "This script must be run from an interactive PowerShell terminal."
        exit 1
    }
}

function Invoke-PowerShellCoreTransition {
    <#
    .SYNOPSIS
        Transitions the script execution to PowerShell Core (pwsh) if available.

    .DESCRIPTION
        If the current major version is less than 7 and pwsh is found in the PATH,
        the script relaunches itself using pwsh for better performance and compatibility.

    .EXAMPLE
        Invoke-PowerShellCoreTransition
        Relaunches the script in PowerShell Core if available and version < 7.
    #>
    [CmdletBinding()]
    param()

    if ($PSVersionTable.PSVersion.Major -lt 7) {
        $powerShellCoreCommand = Get-Command -Name 'pwsh' -ErrorAction SilentlyContinue
        if ($powerShellCoreCommand) {
            Write-Debug "Relaunching in PowerShell Core for better performance..."
            & $powerShellCoreCommand.Source -NoProfile -ExecutionPolicy Bypass -File $PSCommandPath @args
            exit $LASTEXITCODE
        }
    }
}

function Write-FormattedStep {
    <#
    .SYNOPSIS
        Outputs a formatted step indicator to the console.

    .DESCRIPTION
        Uses Write-Host with specific colors and formatting to highlight major
        logical steps in the script execution.

    .PARAMETER Message
        The string message to display as a step indicator.

    .EXAMPLE
        Write-FormattedStep "Copying VS Code settings"
        Displays a formatted step message in cyan with bold font weight.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, HelpMessage = "Step message to display")]
        [ValidateNotNullOrEmpty()]
        [string]$Message
    )

    Write-Host -Object "`n==> $Message" -ForegroundColor Cyan -FontWeight Bold
}

# --- Helper Functions ---

function Get-RepositoryRoot {
    <#
    .SYNOPSIS
        Determines the project root directory.

    .DESCRIPTION
        Attempts to find the git repository root using the git command.
        Falls back to the current working directory if git is unavailable.

    .OUTPUTS
        String - The absolute path to the repository root or current working directory.

    .EXAMPLE
        $root = Get-RepositoryRoot
        Returns the repository root path.
    #>
    [CmdletBinding()]
    param()

    [string]$repositoryRoot = $PWD.Path

    $gitCommand = Get-Command -Name 'git' -ErrorAction SilentlyContinue
    if ($gitCommand) {
        try {
            $detectedRoot = (& git rev-parse --show-toplevel 2>$null).Trim()
            if ($detectedRoot -and (Test-Path -LiteralPath $detectedRoot)) {
                Write-Debug "Detected Git repository root: $detectedRoot"
                $repositoryRoot = $detectedRoot
            }
        } catch {
            Write-Debug "Git root detection failed, using current directory"
        }
    }

    return $repositoryRoot
}

function Copy-SettingsTemplate {
    <#
    .SYNOPSIS
        Copies the VS Code settings template to the .vscode directory.

    .DESCRIPTION
        Copies settings.json.template from the template directory to .vscode/settings.json,
        creating the .vscode directory if it does not exist.

    .PARAMETER RepositoryRoot
        The root directory of the repository where the template and .vscode directories are located.

    .EXAMPLE
        Copy-SettingsTemplate -RepositoryRoot "C:\Projects\MyRepo"
        Copies the template to the .vscode directory in the specified repository.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, HelpMessage = "Repository root directory path")]
        [ValidateNotNullOrEmpty()]
        [string]$RepositoryRoot
    )

    $templateFilePath = Join-Path -Path $RepositoryRoot -ChildPath 'template\settings.json.template'
    $vscodeDirectoryPath = Join-Path -Path $RepositoryRoot -ChildPath '.vscode'
    $settingsFilePath = Join-Path -Path $vscodeDirectoryPath -ChildPath 'settings.json'

    if (-not (Test-Path -LiteralPath $templateFilePath)) {
        throw "Template file not found: $templateFilePath"
    }

    $null = New-Item -ItemType Directory -Path $vscodeDirectoryPath -Force

    Write-FormattedStep "Copying VS Code settings template"
    Write-Verbose "Source: $templateFilePath"
    Write-Verbose "Destination: $settingsFilePath"

    Copy-Item -LiteralPath $templateFilePath -Destination $settingsFilePath -Force
}

# --- Main Script Execution ---

Initialize-ScriptEnvironment
Test-IsInteractivePowerShell
Invoke-PowerShellCoreTransition

try {
    Assert-WindowsPlatform
    $repositoryRoot = Get-RepositoryRoot
    Copy-SettingsTemplate -RepositoryRoot $repositoryRoot

    Write-FormattedStep "Success: VS Code settings written to .vscode/settings.json"
} catch {
    Write-Error -Message "Failed to copy VS Code settings template: $($_.Exception.Message)" -ErrorAction Continue
    Write-Debug -Message "Stack Trace: $($_.ScriptStackTrace)"
    exit 1
}
