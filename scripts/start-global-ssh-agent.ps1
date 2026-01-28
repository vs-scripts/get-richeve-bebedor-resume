<# :
@echo off
echo.
echo Error: This script must be run from a PowerShell terminal.
echo.
exit /b 1
#>

<#
.SYNOPSIS
    Configures SSH-Agent service for global access on Windows.

.DESCRIPTION
    This script ensures the SSH-Agent service is installed, configured,
    and running globally on the system. It installs OpenSSH Client if
    needed, configures the service for automatic startup, and sets up
    registry settings for global access.

.NOTES
    Author: Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>
    Version: 0.0.0
    Last Modified: 2026-01-28
    Platform: Windows only
    Requirements: pwsh 7.5.4, Administrator privileges

.EXAMPLE
    .\start-global-ssh-agent.ps1
    Installs and configures SSH-Agent for global system access.

.EXIT CODES
    0 - Success
    1 - Failure (with error message)
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

#region Helper Functions

function Install-OpenSSHClientCapability {
    <#
    .SYNOPSIS
        Installs OpenSSH Client if not already present.

    .DESCRIPTION
        Uses Windows Capability features to install OpenSSH Client,
        which includes the SSH-Agent service.

    .OUTPUTS
        None. Installs OpenSSH Client capability.

    .EXAMPLE
        Install-OpenSSHClientCapability
        Installs OpenSSH Client on the system.
    #>
    [CmdletBinding()]
    param()

    $capability = Get-WindowsCapability -Online | `
        Where-Object { $_.Name -like 'OpenSSH.Client*' }

    if ($capability) {
        Write-DebugLog -Scope "SSH-INSTALL" `
            -Message "Installing OpenSSH Client"

        Add-WindowsCapability -Online -Name $capability.Name

        Write-InfoLog -Scope "SSH-INSTALL" `
            -Message "OpenSSH Client installed successfully"
    } else {
        throw "OpenSSH Client capability not found"
    }
}

function Install-SSHAgentServiceIfNotPresent {
    <#
    .SYNOPSIS
        Ensures SSH-Agent service is installed and available.

    .DESCRIPTION
        Checks for SSH-Agent service, installs OpenSSH Client if needed,
        and returns the service object.

    .OUTPUTS
        System.ServiceProcess.ServiceController - SSH-Agent service.

    .EXAMPLE
        $service = Install-SSHAgentServiceIfNotPresent
        Ensures SSH-Agent service is available.
    #>
    [CmdletBinding()]
    param()

    $service = Get-Service -Name "ssh-agent" `
        -ErrorAction SilentlyContinue

    if (-not $service) {
        Write-InfoLog -Scope "SSH-SERVICE" `
            -Message "SSH-Agent service not found, installing"

        Install-OpenSSHClientCapability

        # Refresh service list
        $service = Get-Service -Name "ssh-agent" `
            -ErrorAction SilentlyContinue

        if (-not $service) {
            throw "SSH-Agent service still not found after installation"
        }
    }

    Write-InfoLog -Scope "SSH-SERVICE" `
        -Message "SSH-Agent service found"

    return $service
}

function Set-SSHAgentForGlobalAccess {
    <#
    .SYNOPSIS
        Configures SSH-Agent service for global access.

    .DESCRIPTION
        Sets the SSH-Agent service startup type to Automatic and configures
        registry settings for global access. Registry configuration is
        optional and the service will work without it.

    .OUTPUTS
        None. Configures SSH-Agent service and registry settings.

    .EXAMPLE
        Set-SSHAgentForGlobalAccess
        Configures SSH-Agent for global system access.
    #>
    [CmdletBinding()]
    param()

    Write-InfoLog -Scope "SSH-CONFIG" `
        -Message "Configuring SSH-Agent for global access"

    # Set service startup type to Automatic
    Set-Service -Name "ssh-agent" -StartupType Automatic `
        -ErrorAction Stop

    Write-InfoLog -Scope "SSH-CONFIG" `
        -Message "SSH-Agent service set to start automatically"

    # Configure registry for global SSH agent access
    try {
        $registryPath = "HKLM:\SOFTWARE\OpenSSH"
        if (-not (Test-Path -LiteralPath $registryPath)) {
            New-Item -Path $registryPath -Force | Out-Null
        }

        # Set agent to be available for all users
        Set-ItemProperty -Path $registryPath -Name "Agent" `
            -Value "1" -Type String -Force

        Write-InfoLog -Scope "SSH-CONFIG" `
            -Message "Registry settings configured for global access"
    } catch {
        Write-WarningLog -Scope "SSH-CONFIG" `
            -Message "Failed to configure registry: $($_.Exception.Message)"

        Write-InfoLog -Scope "SSH-CONFIG" `
            -Message "Service should still work without registry settings"
    }
}

function Start-SSHAgentServiceAndVerify {
    <#
    .SYNOPSIS
        Starts the SSH-Agent service and verifies it is running.

    .DESCRIPTION
        Starts the SSH-Agent service if not already running and verifies
        the service status.

    .OUTPUTS
        System.ServiceProcess.ServiceController - SSH-Agent service.

    .EXAMPLE
        $service = Start-SSHAgentServiceAndVerify
        Starts SSH-Agent and returns the service object.
    #>
    [CmdletBinding()]
    param()

    $service = Get-Service -Name "ssh-agent"

    if ($service.Status -ne 'Running') {
        Write-InfoLog -Scope "SSH-START" `
            -Message "Starting SSH-Agent service"

        Start-Service -Name "ssh-agent" -ErrorAction Stop

        Write-InfoLog -Scope "SSH-START" `
            -Message "SSH-Agent service started successfully"
    } else {
        Write-InfoLog -Scope "SSH-START" `
            -Message "SSH-Agent service is already running"
    }

    # Verify service status
    $serviceStatus = Get-Service -Name "ssh-agent"
    if ($serviceStatus.Status -eq 'Running') {
        Write-InfoLog -Scope "SSH-VERIFY" `
            -Message "SSH-Agent service running and configured"

        Write-DebugLog -Scope "SSH-VERIFY" `
            -Message "Service startup type: $($serviceStatus.StartType)"

        Write-DebugLog -Scope "SSH-VERIFY" `
            -Message "Service status: $($serviceStatus.Status)"
    } else {
        $statusMsg = "SSH-Agent failed to start. Status: " +
            "$($serviceStatus.Status)"
        throw $statusMsg
    }

    return $serviceStatus
}

#endregion

#region Primary Functions

function Invoke-SSHAgentConfigurationWorkflow {
    <#
    .SYNOPSIS
        Executes the full workflow to configure SSH-Agent globally.

    .DESCRIPTION
        Orchestrates OpenSSH installation, service configuration, and
        startup to enable global SSH-Agent access on the system.

    .OUTPUTS
        None. Configures SSH-Agent for global access.

    .EXAMPLE
        Invoke-SSHAgentConfigurationWorkflow
        Configures SSH-Agent for global system access.
    #>
    [CmdletBinding()]
    param()

    Write-InfoLog -Scope "SSH-WORKFLOW" `
        -Message "Initializing Global SSH-Agent Configuration"

    # 1. Ensure SSH-Agent service is available
    $null = Install-SSHAgentServiceIfNotPresent

    # 2. Configure service for global access
    Set-SSHAgentForGlobalAccess

    # 3. Start and verify the service
    Start-SSHAgentServiceAndVerify

    Write-InfoLog -Scope "SSH-WORKFLOW" `
        -Message "Success: SSH-Agent configured for global access"

    Write-InfoLog -Scope "SSH-WORKFLOW" `
        -Message "SSH-Agent will start automatically on system boot"

    Write-InfoLog -Scope "SSH-WORKFLOW" `
        -Message "Users can now use 'ssh-add' to manage SSH keys"
}

#endregion

#region Main Script Execution

Initialize-ScriptEnvironment
Assert-WindowsPlatform
Assert-PowerShellVersionStrict

# Check for elevation (REQUIRED for elevated scripts)
if (-not (Test-IsAdministrator)) {
    Invoke-ElevationRequest
}

try {
    Invoke-SSHAgentConfigurationWorkflow

    # Final status check
    $finalStatus = Get-Service -Name "ssh-agent" `
        -ErrorAction SilentlyContinue

    if ($finalStatus) {
        Write-DebugLog -Scope "SSH-FINAL" `
            -Message "Final status: $($finalStatus.Status)"

        Write-DebugLog -Scope "SSH-FINAL" `
            -Message "Startup: $($finalStatus.StartType)"
    } else {
        Write-WarningLog -Scope "SSH-FINAL" `
            -Message "Could not verify final service status"
    }

    exit 0
} catch {
    Write-ErrorLog -Scope "SSH-MAIN" `
        -Message "Failed to configure SSH-Agent: $($_.Exception.Message)"

    Write-DebugLog -Scope "SSH-MAIN" `
        -Message "Stack Trace: $($_.ScriptStackTrace)"

    exit 1
}

#endregion
