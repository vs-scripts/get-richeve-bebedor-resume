@{
    RootModule           = 'concise-log.psm1'
    ModuleVersion        = '0.0.0'
    GUID                 = '00000000-0000-0000-0000-000000000000'
    Author               = 'Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>'
    CompanyName          = 'vs-scripts'
    Copyright            = '(c) 2026 Richeve Bebedor. All rights reserved.'
    Description          = 'Provides logging functions for concise log format'
    PowerShellVersion    = '7.5.4'
    FunctionsToExport    = @(
        'Write-Log'
        'Write-DebugLog'
        'Write-InfoLog'
        'Write-WarningLog'
        'Write-ErrorLog'
        'Write-ExceptionLog'
    )
    PrivateData          = @{
        PSData           = @{
            Tags         = @('logging', 'concise-log', 'module')
            ProjectUri   = 'https://github.com/vs-scripts/install-node-version'
            LicenseUri   = 'https://github.com/vs-scripts/install-node-version/LICENSE'
        }
    }
}
