@{
    RootModule           = 'powershell-core.psm1'
    ModuleVersion        = '0.0.0'
    GUID                 = '00000000-0000-0000-0000-000000000000'
    Author               = 'Richeve Bebedor'
    CompanyName          = 'vs-scripts'
    Copyright            = '(c) 2026 Richeve Bebedor. All rights reserved.'
    Description          = 'Core functions for elevated and non-elevated executable PowerShell scripts.'
    PowerShellVersion    = '7.5.4'
    FunctionsToExport    = @(
        'Initialize-ScriptEnvironment'
        'Assert-WindowsPlatform'
        'Test-IsInteractivePowerShell'
        'Invoke-PowerShellCoreTransition'
        'Assert-PowerShellVersionStrict'
        'Test-IsAdministrator'
        'Invoke-ElevationRequest'
    )
    PrivateData          = @{
        PSData           = @{
            Tags         = @('core', 'elevation', 'utilities', 'powershell')
            ProjectUri   = 'https://github.com/vs-scripts/install-node-version'
            LicenseUri   = 'https://github.com/vs-scripts/install-node-version/LICENSE'
        }
    }
}
