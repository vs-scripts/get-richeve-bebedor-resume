<#
.SYNOPSIS
    Provides logging functions for generating concise log entries.

.DESCRIPTION
    This module implements the concise log format specification for
    generating standardized log entries. It includes functions for
    logging at different levels (Debug, Information, Warning, Error,
    Exception) and ensures all log entries follow the CLOGF
    specification.

.NOTES
    Author: Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>
    Version: 0.0.0
    Last Modified: 2026-01-21
    Platform: Windows only
    Requirements: pwsh 7.5.4

.EXAMPLE
    # Import-Module -Name concise-log
    Write-Log -Level "I" -Scope "DATA-ACCOUNTS" -Message "Cannot add account data"

.EXIT CODES
    0 - Success
    1 - Failure (with error message)
#>

Set-StrictMode -Version Latest

#region Configuration

# Module-level configuration variables
$script:DisableLogColors = $false
$script:EnableVerboseMode = $false

#endregion

#region Public Functions

function Write-Log {
    <#
    .SYNOPSIS
        Writes a log entry in the concise log format.

    .DESCRIPTION
        Generates a log entry following the concise log format
        specification. The log entry includes a timestamp, log level,
        scope, message, and reference. Output method depends on log
        level and verbose mode setting.

    .PARAMETER Level
        The log level (D, I, W, E, X).

    .PARAMETER Scope
        The log scope in PARENT-CHILD format.

    .PARAMETER Message
        The log message.

    .PARAMETER PassThru
        If specified, returns the formatted log entry.

    .OUTPUTS
        [string] The formatted log entry if PassThru is specified.

    .EXAMPLE
        # Writes an information log entry.
        Write-Log -Level "I" -Scope "DATA-ACCOUNTS" `
            -Message "Cannot add account data"

    .NOTES
        This function applies line wrapping to ensure log entries do
        not exceed 83 characters per line.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('D', 'I', 'W', 'E', 'X')]
        [string]$Level,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Scope,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Message,

        [Parameter(Mandatory = $false)]
        [switch]$PassThru
    )

    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss.ffZ"
    $logEntry = "# $timestamp $Level $Scope $Message"

    # Generate reference
    $hash = Get-LogHash -LogEntry $logEntry
    $scopeParts = $Scope.Split('-')

    # Validate PARENT-CHILD format
    if ($scopeParts.Length -lt 2) {
        Write-Warning "Scope '$Scope' should be in PARENT-CHILD format"
        $scopeParts = @($Scope, 'GENERAL')
    }

    $reference = "urn:cla:$($scopeParts[0].ToLower()):" +
        "$($scopeParts[1].ToLower()):$hash"

    # Format the log entry
    $formattedLog = "# $timestamp $Level $Scope $Message $reference"

    # Apply line wrapping if the log entry exceeds 83 characters
    if ($formattedLog.Length -gt 83) {
        # Split the log entry into lines, preserving the header
        $header = "# $timestamp $Level $Scope"
        $remainingMessage = " $Message $reference"

        # Calculate available space for the message after the header
        $headerLength = $header.Length + 1
        $availableSpace = 83 - $headerLength

        # Split the remaining message into chunks that fit within the
        # available space. Break at word boundaries (spaces) instead of
        # cutting words.
        $messageChunks = @()
        $remainingText = $remainingMessage.TrimStart()

        while ($remainingText.Length -gt 0) {
            if ($remainingText.Length -le $availableSpace) {
                $messageChunks += $remainingText
                $remainingText = ""
            } else {
                # Find the last space within the available space
                $chunk = $remainingText.Substring(0, $availableSpace)
                $lastSpaceIndex = $chunk.LastIndexOf(' ')

                if ($lastSpaceIndex -gt 0) {
                    # Break at the last space
                    $chunk = $chunk.Substring(0, $lastSpaceIndex)
                    $messageChunks += $chunk
                    $remainingText = $remainingText.Substring(
                        $lastSpaceIndex + 1
                    )
                } else {
                    # No space found, break at available space
                    $messageChunks += $chunk
                    $remainingText = $remainingText.Substring(
                        $availableSpace
                    )
                }
            }
        }

        # Construct the formatted log with line wrapping
        $formattedLog = $header + " " + $messageChunks[0]
        for ($index = 1; $index -lt $messageChunks.Length; $index++) {
            $formattedLog += "`n" + $header + " " + `
                $messageChunks[$index]
        }
    }

    # Output based on log level and verbose mode
    switch ($Level) {
        'D' {
            # Debug level always uses built-in Write-Debug
            Write-Debug $formattedLog
        }
        'I' {
            if ($script:EnableVerboseMode) {
                Write-Verbose $formattedLog
            } else {
                if ($script:DisableLogColors) {
                    Write-Host $formattedLog
                } else {
                    Write-Host $formattedLog -ForegroundColor DarkGreen
                }
            }
        }
        'W' {
            if ($script:EnableVerboseMode) {
                Write-Verbose $formattedLog
            } else {
                if ($script:DisableLogColors) {
                    Write-Host $formattedLog
                } else {
                    Write-Host $formattedLog -ForegroundColor DarkYellow
                }
            }
        }
        'E' {
            if ($script:EnableVerboseMode) {
                Write-Verbose $formattedLog
            } else {
                if ($script:DisableLogColors) {
                    Write-Host $formattedLog
                } else {
                    Write-Host $formattedLog -ForegroundColor Red
                }
            }
        }
        'X' {
            if ($script:EnableVerboseMode) {
                Write-Verbose $formattedLog
            } else {
                if ($script:DisableLogColors) {
                    Write-Host $formattedLog
                } else {
                    Write-Host $formattedLog -ForegroundColor DarkRed
                }
            }
        }
    }

    if ($PassThru) {
        $formattedLog
    }
}

function Write-DebugLog {
    <#
    .SYNOPSIS
        Writes a debug log entry.

    .DESCRIPTION
        Writes a log entry at the Debug level.

    .PARAMETER Scope
        The log scope in PARENT-CHILD format.

    .PARAMETER Message
        The log message.

    .PARAMETER PassThru
        If specified, returns the formatted log entry.

    .OUTPUTS
        [string] The formatted log entry if PassThru is specified.

    .EXAMPLE
        # Writes a debug log entry.
        Write-DebugLog -Scope "DATA-ACCOUNTS" -Message "Debugging account data"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Scope,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Message,

        [Parameter(Mandatory = $false)]
        [switch]$PassThru
    )

    Write-Log -Level "D" -Scope $Scope -Message $Message -PassThru:$PassThru
}

function Write-InfoLog {
    <#
    .SYNOPSIS
        Writes an information log entry.

    .DESCRIPTION
        Writes a log entry at the Information level.

    .PARAMETER Scope
        The log scope in PARENT-CHILD format.

    .PARAMETER Message
        The log message.

    .PARAMETER PassThru
        If specified, returns the formatted log entry.

    .OUTPUTS
        [string] The formatted log entry if PassThru is specified.

    .EXAMPLE
        # Writes an information log entry.
        Write-InfoLog -Scope "DATA-ACCOUNTS" `
            -Message "Processing account data"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Scope,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Message,

        [Parameter(Mandatory = $false)]
        [switch]$PassThru
    )

    Write-Log `
        -Level "I" `
        -Scope $Scope `
        -Message $Message `
        -PassThru:$PassThru
}

function Write-WarningLog {
    <#
    .SYNOPSIS
        Writes a warning log entry.

    .DESCRIPTION
        Writes a log entry at the Warning level.

    .PARAMETER Scope
        The log scope in PARENT-CHILD format.

    .PARAMETER Message
        The log message.

    .PARAMETER PassThru
        If specified, returns the formatted log entry.

    .OUTPUTS
        [string] The formatted log entry if PassThru is specified.

    .EXAMPLE
        # Writes a warning log entry.
        Write-WarningLog -Scope "DATA-ACCOUNTS" `
            -Message "Account data may be incomplete"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Scope,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Message,

        [Parameter(Mandatory = $false)]
        [switch]$PassThru
    )

    Write-Log `
        -Level "W" `
        -Scope $Scope `
        -Message $Message `
        -PassThru:$PassThru
}

function Write-ErrorLog {
    <#
    .SYNOPSIS
        Writes an error log entry.

    .DESCRIPTION
        Writes a log entry at the Error level.

    .PARAMETER Scope
        The log scope in PARENT-CHILD format.

    .PARAMETER Message
        The log message.

    .PARAMETER PassThru
        If specified, returns the formatted log entry.

    .OUTPUTS
        [string] The formatted log entry if PassThru is specified.

    .EXAMPLE
        # Writes an error log entry.
        Write-ErrorLog -Scope "DATA-ACCOUNTS" `
            -Message "Failed to add account data"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Scope,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Message,

        [Parameter(Mandatory = $false)]
        [switch]$PassThru
    )

    Write-Log `
        -Level "E" `
        -Scope $Scope `
        -Message $Message `
        -PassThru:$PassThru
}

function Write-ExceptionLog {
    <#
    .SYNOPSIS
        Writes an exception log entry.

    .DESCRIPTION
        Writes a log entry at the Exception level.

    .PARAMETER Scope
        The log scope in PARENT-CHILD format.

    .PARAMETER Message
        The log message.

    .PARAMETER PassThru
        If specified, returns the formatted log entry.

    .OUTPUTS
        [string] The formatted log entry if PassThru is specified.

    .EXAMPLE
        # Writes an exception log entry.
        Write-ExceptionLog -Scope "DATA-ACCOUNTS" `
            -Message "Unexpected error in account data"
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Scope,

        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$Message,

        [Parameter(Mandatory = $false)]
        [switch]$PassThru
    )

    Write-Log `
        -Level "X" `
        -Scope $Scope `
        -Message $Message `
        -PassThru:$PassThru
}

#endregion

#region Private Functions

function Get-LogHash {
    <#
    .SYNOPSIS
        Generates a hash for the log entry.

    .DESCRIPTION
        Computes a 5-character alphanumeric hash of the log entry for
        use as a reference.

    .PARAMETER LogEntry
        The log entry to hash.

    .OUTPUTS
        [string] A 5-character alphanumeric hash.

    .EXAMPLE
        # Returns a 5-character hash.
        $hash = Get-LogHash -LogEntry `
            "# 2024-01-15T05:55:00.00Z I DATA-ACCOUNTS Cannot add"

    .NOTES
        This is a private function and is not exported.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$LogEntry
    )

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($LogEntry)

    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hashBytes = $sha256.ComputeHash($bytes)
    } finally {
        $sha256.Dispose()
    }

    $hash = [System.BitConverter]::ToString($hashBytes).Replace(
        "-", ""
    ).ToLower()

    return $hash.Substring(0, 5)
}

#endregion

# Export public functions
Export-ModuleMember -Function @(
    'Write-Log'
    'Write-DebugLog'
    'Write-InfoLog'
    'Write-WarningLog'
    'Write-ErrorLog'
    'Write-ExceptionLog'
)
