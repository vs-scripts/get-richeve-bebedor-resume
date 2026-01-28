<# :
@echo off
echo.
echo Error: This script must be run from a PowerShell terminal.
echo.
exit /b 1
#>

<#
.SYNOPSIS
    Installs and configures Volta to pin Node.js LTS to the repository.

.DESCRIPTION
    This script ensures Volta is installed on the system, adds it to the PATH,
    creates a package.json if needed, and pins the latest Node.js LTS version
    to the repository. Volta manages Node.js versions per-project.

.NOTES
    Author: Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>
    Version: 0.0.0
    Platform: Windows only
    Requirements: pwsh 7.5.4, Administrator privileges

.EXAMPLE
    .\install-node-version.ps1
    Installs Volta and pins Node.js LTS to the current repository.

.EXIT CODES
    0 - Success
    1 - Failure (with error message)
#>

[CmdletBinding()]
param()

#region Module Imports

# Import required modules
$scriptPath = $PSScriptRoot
if (-not $scriptPath) {
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
}

if (-not $scriptPath) {
    Write-Host "FATAL: Cannot determine script directory" -ForegroundColor Red
    Write-Host "PSScriptRoot: $PSScriptRoot" -ForegroundColor Yellow

    $commandPath = $MyInvocation.MyCommand.Path
    Write-Host "MyInvocation.MyCommand.Path: $commandPath" `
        -ForegroundColor Yellow

    exit 1
}

$conciseLogPath = Join-Path $scriptPath 'concise-log.psm1'
$coreModulePath = Join-Path $scriptPath 'powershell-core.psm1'

# Convert module paths to absolute paths
$conciseLogPath = [System.IO.Path]::GetFullPath($conciseLogPath)
$coreModulePath = [System.IO.Path]::GetFullPath($coreModulePath)

if (-not (Test-Path -LiteralPath $conciseLogPath)) {
    Write-Host "ERROR: Required module not found: $conciseLogPath" `
        -ForegroundColor Red
    Write-Host "Script directory: $scriptPath" -ForegroundColor Yellow
    Write-Host "Current location: $PWD" -ForegroundColor Yellow
    Write-Host "Files in script directory:" -ForegroundColor Yellow
    if (Test-Path -LiteralPath $scriptPath) {
        Get-ChildItem -LiteralPath $scriptPath -Filter "*.psm1" |
            ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Cyan }
    }
    exit 1
}

if (-not (Test-Path -LiteralPath $coreModulePath)) {
    Write-Host "ERROR: Required module not found: $coreModulePath" `
        -ForegroundColor Red
    Write-Host "Script directory: $scriptPath" -ForegroundColor Yellow
    Write-Host "Current location: $PWD" -ForegroundColor Yellow
    exit 1
}

try {
    Import-Module -Name $conciseLogPath -Force -ErrorAction Stop -Global
    Import-Module -Name $coreModulePath -Force -ErrorAction Stop -Global
} catch {
    Write-Host "ERROR: Failed to import modules: $_" -ForegroundColor Red
    Write-Host "concise-log path: $conciseLogPath" -ForegroundColor Yellow
    Write-Host "powershell-core path: $coreModulePath" -ForegroundColor Yellow

    $exceptionType = $_.Exception.GetType().FullName
    Write-Host "Exception: $exceptionType" -ForegroundColor Yellow

    Write-Host "Message: $($_.Exception.Message)" -ForegroundColor Yellow
    exit 1
}

# Verify modules loaded
$conciseLogLoaded = Get-Module -Name 'concise-log' `
    -ErrorAction SilentlyContinue
$coreModuleLoaded = Get-Module -Name 'powershell-core' `
    -ErrorAction SilentlyContinue

if (-not $conciseLogLoaded) {
    Write-Host "ERROR: concise-log module not loaded" -ForegroundColor Red
    exit 1
}

if (-not $coreModuleLoaded) {
    Write-Host "ERROR: powershell-core module not loaded" -ForegroundColor Red
    exit 1
}

#endregion

# Initialize environment (sets StrictMode and preferences)
Initialize-ScriptEnvironment

#region Helper Functions

function Get-RepositoryRoot {
    <#
    .SYNOPSIS
        Determines the project root directory.

    .DESCRIPTION
        Attempts to find the git repository root using the git command.
        Falls back to the current working directory if git is unavailable.

    .OUTPUTS
        String - The absolute path to the repository root or current working
                    directory.

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
                $null = Write-DebugLog -Scope "REPO-ROOT" `
                    -Message "Detected Git repository root: $detectedRoot"

                $repositoryRoot = $detectedRoot
            }
        } catch {
            $null = Write-DebugLog -Scope "REPO-ROOT" `
                -Message "Git root detection failed, using current directory"
        }
    }

    return $repositoryRoot
}

#endregion

#region Package Management Functions

function Install-PackageWithWinget {
    <#
    .SYNOPSIS
        Installs a package using the Windows Package Manager (winget).

    .DESCRIPTION
        Standardizes the winget installation command with agreements and
        silent flags. Throws if winget is missing.

    .PARAMETER PackageIdentifier
        The ID of the package to install (e.g., "Volta.Volta").

    .EXAMPLE
        Install-PackageWithWinget -PackageIdentifier "Volta.Volta"
        Installs Volta using winget.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, `
            HelpMessage = "Package identifier for winget")]
        [ValidateNotNullOrEmpty()]
        [string]$PackageIdentifier
    )

    $wingetCommand = Get-Command -Name 'winget' -ErrorAction SilentlyContinue
    if (-not $wingetCommand) {
        Write-ErrorLog -Scope "WINGET-INSTALL" `
            -Message "winget not found for package $PackageIdentifier"

        $failureMessage = "Package '$PackageIdentifier' is not installed " +
            "and winget was not found."

        throw $failureMessage
    }

    Write-DebugLog -Scope "WINGET-INSTALL" `
        -Message "Installing package $PackageIdentifier via winget"

    & winget install `
        --id $PackageIdentifier `
        --source winget `
        --silent `
        --accept-package-agreements `
        --accept-source-agreements

    $allowedExitCodes = @(0, -1978335189)
    if ($LASTEXITCODE -notin $allowedExitCodes) {
        $warningMessage = "winget install failed for $PackageIdentifier " +
            "(exit $LASTEXITCODE)"

        Write-WarningLog -Scope "WINGET-INSTALL" `
            -Message $warningMessage

        Write-ErrorLog -Scope "WINGET-INSTALL" `
            -Message "winget install failed; aborting"

        throw $warningMessage
    } elseif ($LASTEXITCODE -ne 0) {
        $warningMessage = "winget reported no applicable upgrade for " +
            "$PackageIdentifier (exit $LASTEXITCODE)"
        Write-WarningLog -Scope "WINGET-INSTALL" `
            -Message $warningMessage
    }
}

#endregion

#region Volta Management Functions

function Install-VoltaIfMissing {
    <#
    .SYNOPSIS
        Installs Volta if it is not already available on the system.

    .DESCRIPTION
        Checks for the 'volta' command. If missing, attempts to install it
            via winget.

    .EXAMPLE
        Install-VoltaIfMissing
        Installs Volta if not already present.
    #>
    [CmdletBinding()]
    param()

    Add-VoltaToSessionPath

    $voltaCommand = Get-Command -Name 'volta' -ErrorAction SilentlyContinue
    if ($voltaCommand) {
        Write-InfoLog -Scope "VOLTA-INSTALL" `
            -Message "Volta already installed at $($voltaCommand.Source)"

        return
    }

    Write-InfoLog -Scope "VOLTA-INSTALL" `
        -Message "Volta not found. Installing via winget."

    Install-PackageWithWinget -PackageIdentifier "Volta.Volta"

    Add-VoltaToSessionPath

    $voltaCommand = Get-Command -Name 'volta' -ErrorAction SilentlyContinue
    if (-not $voltaCommand) {
        Write-ErrorLog -Scope "VOLTA-INSTALL" `
            -Message "Volta not found after installation"

        throw "Volta installation failed or not on PATH"
    }
}

function Add-VoltaToSessionPath {
    <#
    .SYNOPSIS
        Ensures the Volta binary directory is in the current session's PATH.

    .DESCRIPTION
        Checks if the Volta bin directory exists and adds it to the session PATH
        if not already present. Uses case-insensitive comparison for path matching.

    .EXAMPLE
        # Adds Volta bin directory to the current session PATH.
        Add-VoltaToSessionPath
    #>
    [CmdletBinding()]
    param()

    $voltaBinaryDirectory = Join-Path -Path $env:LOCALAPPDATA `
        -ChildPath 'Volta\bin'

    if (-not (Test-Path -LiteralPath $voltaBinaryDirectory)) {
        Write-DebugLog -Scope "VOLTA-PATH" `
            -Message "Creating Volta bin directory: $voltaBinaryDirectory"

        New-Item -ItemType Directory -Path $voltaBinaryDirectory -Force | Out-Null
    }

    $voltaDirectories = @($voltaBinaryDirectory)
    if ($env:ProgramFiles) {
        $voltaProgramFiles = Join-Path -Path $env:ProgramFiles `
            -ChildPath 'Volta'
        if (Test-Path -LiteralPath $voltaProgramFiles) {
            $voltaDirectories += $voltaProgramFiles
        }
    }
    if (${env:ProgramFiles(x86)}) {
        $voltaProgramFilesX86 = Join-Path `
            -Path ${env:ProgramFiles(x86)} `
            -ChildPath 'Volta'
        if (Test-Path -LiteralPath $voltaProgramFilesX86) {
            $voltaDirectories += $voltaProgramFilesX86
        }
    }

    $pathSeparator = [System.IO.Path]::PathSeparator
    $pathEntries = ($env:PATH -split $pathSeparator) | `
        Where-Object { $_ -ne '' }

    foreach ($voltaDirectory in $voltaDirectories) {
        $isVoltaInPath = $false
        foreach ($pathEntry in $pathEntries) {
            try {
                $normalizedPathEntry = `
                    [System.IO.Path]::GetFullPath($pathEntry).TrimEnd('\')
                $normalizedVoltaPath = `
                    [System.IO.Path]::GetFullPath($voltaDirectory).TrimEnd('\')

                if ($normalizedPathEntry -ieq $normalizedVoltaPath) {
                    $isVoltaInPath = $true
                    break
                }
            } catch {
                Write-DebugLog -Scope "VOLTA-PATH" `
                    -Message "Path normalize error: $_"
            }
        }

        if (-not $isVoltaInPath) {
            Write-InfoLog -Scope "VOLTA-PATH" `
                -Message "Adding Volta directory to PATH: $voltaDirectory"

            $env:PATH = "$voltaDirectory$pathSeparator$env:PATH"
            $pathEntries = ($env:PATH -split $pathSeparator) | `
                Where-Object { $_ -ne '' }
        }
    }
}

function Initialize-PackageJsonIfMissing {
    <#
    .SYNOPSIS
        Ensures a package.json file exists for Volta pinning.

    .DESCRIPTION
        Volta requires package.json to pin versions. If missing,
        creates a minimal one with the directory name as the project name.

    .PARAMETER RepositoryRoot
        The directory where package.json should reside.

    .OUTPUTS
        String - The full path to the package.json file.

    .EXAMPLE
        # Creates or returns the path to package.json.
        $packageJsonPath = Initialize-PackageJsonIfMissing `
            -RepositoryRoot "C:\Projects\MyRepo"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, `
            HelpMessage = "Repository root directory path")]
        [ValidateNotNullOrEmpty()]
        [string]$RepositoryRoot
    )

    $packageJsonPath = Join-Path -Path $RepositoryRoot -ChildPath 'package.json'

    if (Test-Path -LiteralPath $packageJsonPath) {
        Write-InfoLog -Scope "PACKAGE-JSON" `
            -Message "Found existing package.json at $packageJsonPath"

        return $packageJsonPath
    }

    Write-InfoLog -Scope "PACKAGE-JSON" `
        -Message "Creating minimal package.json"

    $packageConfiguration = [ordered]@{
        name    = Split-Path -Leaf $RepositoryRoot
        private = $true
    }

    $jsonContent = $packageConfiguration | ConvertTo-Json -Depth 20

    Set-Content -LiteralPath $packageJsonPath -Value $jsonContent -Encoding UTF8

    Write-InfoLog -Scope "PACKAGE-JSON" -Message "Initialized new package.json"

    return $packageJsonPath
}

#endregion

#region Primary Functions

# --- Primary Functions ---

function Invoke-NodeVersionPinningWorkflow {
    <#
    .SYNOPSIS
        Executes the full workflow to install Volta and pin Node.js LTS locally.

    .DESCRIPTION
        Orchestrates Volta installation, PATH updates, package.json initialization,
        and Node.js LTS version pinning for the repository.

    .PARAMETER RepositoryRoot
        The directory context for pinning Node.js.

    .EXAMPLE
        # Installs Volta and pins Node.js LTS to the specified repository.
        Invoke-NodeVersionPinningWorkflow -RepositoryRoot "C:\Projects\MyRepo"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, `
            HelpMessage = "Repository root directory path")]
        [ValidateNotNullOrEmpty()]
        [string]$RepositoryRoot
    )

    Write-InfoLog -Scope "NODE-PIN" `
        -Message "Initializing Node.js LTS environment"

    Write-DebugLog -Scope "NODE-PIN" `
        -Message "Target directory: $RepositoryRoot"

    # 1. Ensure Volta is available
    Install-VoltaIfMissing
    Add-VoltaToSessionPath

    # 2. Ensure package.json exists
    $packageJsonPath = Join-Path -Path $RepositoryRoot -ChildPath 'package.json'
    $normalizeVersion = {
        param(
            [string]$versionOutput,
            [string]$versionLabel
        )

        if (-not $versionOutput) {
            throw "Missing $versionLabel version value"
        }

        $versionValue = $versionOutput.Trim()
        if ($versionValue.StartsWith('v')) {
            $versionValue = $versionValue.Substring(1)
        }

        if (-not ($versionValue -match '^\d+\.\d+\.\d+$')) {
            throw "$versionLabel version is invalid: $versionOutput"
        }

        return $versionValue
    }

    $resolveLtsVersions = {
        & volta install node@lts
        if ($LASTEXITCODE -ne 0) {
            throw "Volta failed to install Node.js LTS"
        }

        $ltsNodeOutput = & volta run --node lts node --version
        $ltsNodeVersion = & $normalizeVersion `
            $ltsNodeOutput `
            "Node.js"

        $ltsNpmOutput = & volta run --node lts --bundled-npm npm --version
        $ltsNpmVersion = & $normalizeVersion `
            $ltsNpmOutput `
            "npm"

        return [ordered]@{
            node = $ltsNodeVersion
            npm  = $ltsNpmVersion
        }
    }

    $targetNodeVersion = $null
    $targetNpmVersion = $null
    $packageData = $null

    if (Test-Path -LiteralPath $packageJsonPath) {
        Write-InfoLog -Scope "PACKAGE-JSON" `
            -Message "Found existing package.json"

        $packageJsonRaw = Get-Content -LiteralPath $packageJsonPath -Raw
        try {
            $packageData = $packageJsonRaw | ConvertFrom-Json
        } catch {
            throw "package.json is not valid JSON"
        }

        $hasEngines = $packageData.PSObject.Properties.Name `
            -contains 'engines'
        $engines = if ($hasEngines) {
            $packageData.engines
        } else {
            $null
        }

        if ($engines -and $engines.node -and $engines.npm) {
            $targetNodeVersion = & $normalizeVersion `
                $engines.node `
                "Node.js"
            $targetNpmVersion = & $normalizeVersion `
                $engines.npm `
                "npm"

            $ltsVersions = & $resolveLtsVersions
            $isNodeLts = $targetNodeVersion -eq $ltsVersions.node
            $isNpmLts = $targetNpmVersion -eq $ltsVersions.npm

            if (-not $isNodeLts -or -not $isNpmLts) {
                Write-WarningLog -Scope "PACKAGE-JSON" `
                    -Message "Engines are not LTS; updating to latest LTS"

                $targetNodeVersion = $ltsVersions.node
                $targetNpmVersion = $ltsVersions.npm

                $enginesValue = [ordered]@{
                    node = $targetNodeVersion
                    npm  = $targetNpmVersion
                }

                $packageData.engines = $enginesValue

                $jsonContent = $packageData | ConvertTo-Json -Depth 20
                Set-Content -LiteralPath $packageJsonPath `
                    -Value $jsonContent `
                    -Encoding UTF8

                Write-InfoLog -Scope "PACKAGE-JSON" `
                    -Message "Updated engines to latest LTS versions"
            } else {
                Write-InfoLog -Scope "PACKAGE-JSON" `
                    -Message "Engines already use latest LTS versions"
            }

            Write-InfoLog -Scope "PACKAGE-JSON" `
                -Message "Using engines from package.json"
        } else {
            Write-InfoLog -Scope "PACKAGE-JSON" `
                -Message "Resolving latest LTS engines"

            $ltsVersions = & $resolveLtsVersions
            $targetNodeVersion = $ltsVersions.node
            $targetNpmVersion = $ltsVersions.npm

            $enginesValue = [ordered]@{
                node = $targetNodeVersion
                npm  = $targetNpmVersion
            }

            if ($hasEngines) {
                $packageData.engines = $enginesValue
            } else {
                $packageData | Add-Member `
                    -NotePropertyName 'engines' `
                    -NotePropertyValue $enginesValue
            }

            $jsonContent = $packageData | ConvertTo-Json -Depth 20
            Set-Content -LiteralPath $packageJsonPath `
                -Value $jsonContent `
                -Encoding UTF8

            Write-InfoLog -Scope "PACKAGE-JSON" `
                -Message "Added engines to package.json"
        }
    } else {
        Write-InfoLog -Scope "PACKAGE-JSON" `
            -Message "Creating minimal package.json"

        $ltsVersions = & $resolveLtsVersions
        $targetNodeVersion = $ltsVersions.node
        $targetNpmVersion = $ltsVersions.npm

        $packageConfiguration = [ordered]@{
            name    = Split-Path -Leaf $RepositoryRoot
            private = $true
            engines = [ordered]@{
                node = $targetNodeVersion
                npm  = $targetNpmVersion
            }
        }

        $jsonContent = $packageConfiguration | ConvertTo-Json -Depth 20
        Set-Content -LiteralPath $packageJsonPath `
            -Value $jsonContent `
            -Encoding UTF8

        Write-InfoLog -Scope "PACKAGE-JSON" `
            -Message "Initialized package.json with engines"

        $packageData = $packageConfiguration
    }

    # 3. Bind Node.js LTS to this folder
    Push-Location -Path $RepositoryRoot
    try {
        $installedNodeVersion = $null
        $installedNpmVersion = $null
        $installedVersionsMatch = $false

        try {
            $installedNodeOutput = & node --version
            $installedNodeVersion = & $normalizeVersion `
                $installedNodeOutput `
                "Node.js"

            $installedNpmOutput = & npm --version
            $installedNpmVersion = & $normalizeVersion `
                $installedNpmOutput `
                "npm"

            $installedVersionsMatch = `
                ($installedNodeVersion -eq $targetNodeVersion) -and `
                ($installedNpmVersion -eq $targetNpmVersion)
        } catch {
            $installedVersionsMatch = $false
        }

        $voltaVersionsMatch = $false
        if ($packageData) {
            $hasVolta = $packageData.PSObject.Properties.Name `
                -contains 'volta'
            $voltaValues = if ($hasVolta) {
                $packageData.volta
            } else {
                $null
            }

            if ($voltaValues -and $voltaValues.node -and $voltaValues.npm) {
                $voltaNodeVersion = & $normalizeVersion `
                    $voltaValues.node `
                    "Node.js"
                $voltaNpmVersion = & $normalizeVersion `
                    $voltaValues.npm `
                    "npm"

                $voltaVersionsMatch = `
                    ($voltaNodeVersion -eq $targetNodeVersion) -and `
                    ($voltaNpmVersion -eq $targetNpmVersion)
            }
        }

        $shouldInstallAndPin = -not ($installedVersionsMatch `
            -and $voltaVersionsMatch)

        if ($shouldInstallAndPin) {
            Write-InfoLog -Scope "NODE-PIN" `
                -Message "Installing Node.js and npm for this folder"

            $installMessage = "Installing Node.js $targetNodeVersion " +
                "and npm $targetNpmVersion"
            Write-InfoLog -Scope "NODE-PIN" -Message $installMessage

            & volta install `
                "node@$targetNodeVersion" `
                "npm@$targetNpmVersion"
            if ($LASTEXITCODE -ne 0) {
                throw "Volta failed to install Node.js or npm"
            }

            Write-InfoLog -Scope "NODE-PIN" `
                -Message "Pinning Node.js and npm to this folder"

            & volta pin `
                "node@$targetNodeVersion" `
                "npm@$targetNpmVersion"
            if ($LASTEXITCODE -ne 0) {
                throw "Volta failed to pin Node.js or npm"
            }

            $installedNodeOutput = & node --version
            $installedNodeVersion = & $normalizeVersion `
                $installedNodeOutput `
                "Node.js"

            $installedNpmOutput = & npm --version
            $installedNpmVersion = & $normalizeVersion `
                $installedNpmOutput `
                "npm"
        } else {
            Write-InfoLog -Scope "NODE-VERIFY" `
                -Message "Versions already match engines; verifying only"
        }

        Write-InfoLog -Scope "NODE-VERIFY" `
            -Message "Validating active Node.js and npm versions"

        if ($installedNodeVersion -ne $targetNodeVersion) {
            $nodeMismatch = "Node.js version mismatch. " +
                "Expected $targetNodeVersion, got $installedNodeVersion"
            throw $nodeMismatch
        }

        if ($installedNpmVersion -ne $targetNpmVersion) {
            $npmMismatch = "npm version mismatch. " +
                "Expected $targetNpmVersion, got $installedNpmVersion"
            throw $npmMismatch
        }

        Write-InfoLog -Scope "NODE-VERIFY" `
            -Message "Validated Node.js and npm versions"
    } finally {
        Pop-Location
    }
}

#endregion

#region Main Script Execution

# --- Main Script Execution ---

Initialize-ScriptEnvironment
$null = Test-IsInteractivePowerShell

Invoke-PowerShellCoreTransition
if (-not (Test-IsAdministrator)) {
    Invoke-ElevationRequest -ScriptPath $PSCommandPath
}

try {
    Assert-WindowsPlatform

    $repositoryRoot = Get-RepositoryRoot

    Invoke-NodeVersionPinningWorkflow -RepositoryRoot $repositoryRoot

    Write-InfoLog -Scope "SCRIPT-MAIN" `
        -Message "Success: Node.js LTS now bound to this folder and session"

    Write-InfoLog -Scope "SCRIPT-MAIN" `
        -Message "Usage: run node in $repositoryRoot"

    exit 0
} catch {
    Write-ErrorLog -Scope "SCRIPT-MAIN" `
        -Message "Failed to install or pin Node.js: $($_.Exception.Message)"

    Write-DebugLog -Scope "SCRIPT-MAIN" `
        -Message "Stack Trace: $($_.ScriptStackTrace)"

    exit 1
}

#endregion
