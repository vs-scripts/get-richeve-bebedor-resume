<# :
@echo off
echo.
echo Error: This script must be run from a PowerShell terminal.
echo.
exit /b 1
#>

<#
.SYNOPSIS
    Validates commit message body format.

.DESCRIPTION
    This script is used as a commit-msg hook to validate that the
    commit message body follows the 5-line format specified in
    .gitmessage template. Validates line count, prefixes, file path
    reference, and content presence.

.NOTES
    Author: Richeve Bebedor <richeve.bebedor+vs-scripts@gmail.com>
    Version: 1.0.0
    Last Modified: 2026-01-28
    Platform: Windows only
    Requirements: pwsh 7.5.4

.EXAMPLE
    # Validates commit message body format
    .\restrict-commit-msg-atomic.ps1 "path/to/commit-msg-file"

.EXIT CODES
    0 - Success (body format valid)
    1 - Failure (body format invalid or hook error)
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$CommitMsgFile
)

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

function Get-CommitMessageBody {
    <#
    .SYNOPSIS
        Extracts non-comment body lines from commit message.

    .DESCRIPTION
        Reads the commit message file, filters out comment lines
        (starting with #), extracts body lines (after header), and
        returns an array of trimmed, non-empty body lines.

    .PARAMETER MessageFilePath
        The path to the commit message file.

    .OUTPUTS
        System.String[]. Array of body lines (non-comment, non-header).

    .EXAMPLE
        $bodyLines = Get-CommitMessageBody -MessageFilePath $file
        Extracts body lines from the commit message.

    .NOTES
        Comment lines and empty lines are filtered out. The first
        non-empty line is considered the header and is skipped.
    #>
    [CmdletBinding()]
    [OutputType([string[]])]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateNotNullOrEmpty()]
        [string]$MessageFilePath
    )

    try {
        Write-DebugLog -Scope "HOOK-COMMITMSG" `
            -Message "Extracting commit message body"

        # Read entire file content
        $rawContent = Get-Content -LiteralPath $MessageFilePath -Raw

        # Split into lines and filter out comments
        $allLines = $rawContent -split "`r?`n"
        $nonCommentLines = $allLines | Where-Object {
            $_ -notmatch '^\s*#'
        }

        # Trim whitespace and filter empty lines
        $nonEmptyLines = $nonCommentLines | ForEach-Object {
            $_.Trim()
        } | Where-Object {
            $_ -ne ''
        }

        # Skip the first line (header) and return body lines
        if ($nonEmptyLines.Count -gt 1) {
            $bodyLines = $nonEmptyLines | Select-Object -Skip 1
            Write-DebugLog -Scope "HOOK-COMMITMSG" `
                -Message "Found $($bodyLines.Count) body lines"
            return $bodyLines
        }
        else {
            Write-DebugLog -Scope "HOOK-COMMITMSG" `
                -Message "No body lines found"
            return @()
        }
    }
    catch {
        Write-ErrorLog -Scope "HOOK-COMMITMSG" `
            -Message "Error reading message: $($_.Exception.Message)"
        throw
    }
}

function Test-BodyLineCount {
    <#
    .SYNOPSIS
        Validates that body contains exactly 5 lines.

    .DESCRIPTION
        Checks if the body lines array contains exactly 5 lines as
        required by the commit message template.

    .PARAMETER BodyLines
        Array of body lines from the commit message.

    .OUTPUTS
        PSCustomObject with IsValid (bool) and ErrorMessage (string).

    .EXAMPLE
        $result = Test-BodyLineCount -BodyLines $lines
        Validates the body line count.

    .NOTES
        Returns IsValid=$true if count is exactly 5, otherwise
        IsValid=$false with descriptive error message.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$BodyLines
    )

    $lineCount = $BodyLines.Count

    Write-DebugLog -Scope "HOOK-COMMITMSG" `
        -Message "Validating line count: $lineCount"

    if ($lineCount -ne 5) {
        return [PSCustomObject]@{
            IsValid = $false
            ErrorMessage = "Body must have exactly 5 lines, found $lineCount"
        }
    }

    return [PSCustomObject]@{
        IsValid = $true
        ErrorMessage = ''
    }
}

function Test-BodyLinePrefixes {
    <#
    .SYNOPSIS
        Validates each line has the correct numbered prefix.

    .DESCRIPTION
        Checks if each of the 5 body lines starts with the correct
        prefix: "1. file:", "2. change:", "3. reason:", "4. impact:",
        "5. verify:". Uses case-insensitive comparison.

    .PARAMETER BodyLines
        Array of body lines from the commit message.

    .OUTPUTS
        PSCustomObject with IsValid (bool) and ErrorMessage (string).

    .EXAMPLE
        $result = Test-BodyLinePrefixes -BodyLines $lines
        Validates the line prefixes.

    .NOTES
        Returns IsValid=$true if all prefixes match, otherwise
        IsValid=$false with line number and expected prefix.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateCount(5, 5)]
        [string[]]$BodyLines
    )

    Write-DebugLog -Scope "HOOK-COMMITMSG" `
        -Message "Validating line prefixes"

    $expectedPrefixes = @(
        '1. file:',
        '2. change:',
        '3. reason:',
        '4. impact:',
        '5. verify:'
    )

    for ($index = 0; $index -lt 5; $index++) {
        $line = $BodyLines[$index]
        $expectedPrefix = $expectedPrefixes[$index]
        $lineNumber = $index + 1

        # Extract actual prefix (up to and including colon)
        if ($line -match '^(\d+\.\s*\w+:)') {
            $actualPrefix = $matches[1]
        }
        else {
            return [PSCustomObject]@{
                IsValid = $false
                ErrorMessage = "Line $lineNumber must start with " +
                    """$expectedPrefix"", found ""$line"""
            }
        }

        # Case-insensitive comparison
        if ($actualPrefix -ne $expectedPrefix -and
            $actualPrefix.ToLower() -ne $expectedPrefix.ToLower()) {
            return [PSCustomObject]@{
                IsValid = $false
                ErrorMessage = "Line $lineNumber must start with " +
                    """$expectedPrefix"", found ""$actualPrefix"""
            }
        }
    }

    return [PSCustomObject]@{
        IsValid = $true
        ErrorMessage = ''
    }
}

function Test-FilePathReference {
    <#
    .SYNOPSIS
        Validates line 1 references the staged file.

    .DESCRIPTION
        Extracts the file path from line 1 (after "1. file:" prefix),
        gets the staged file path from git, normalizes both paths, and
        compares them for a match.

    .PARAMETER BodyLines
        Array of body lines from the commit message.

    .OUTPUTS
        PSCustomObject with IsValid (bool) and ErrorMessage (string).

    .EXAMPLE
        $result = Test-FilePathReference -BodyLines $lines
        Validates the file path reference.

    .NOTES
        Paths are normalized (backslashes to forward slashes) and
        compared case-insensitively.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateCount(5, 5)]
        [string[]]$BodyLines
    )

    try {
        Write-DebugLog -Scope "HOOK-COMMITMSG" `
            -Message "Validating file path reference"

        # Extract file path from line 1
        $line1 = $BodyLines[0]
        if ($line1 -match '^1\.\s*file:\s*(.+)$') {
            $referencedPath = $matches[1].Trim()
        }
        else {
            return [PSCustomObject]@{
                IsValid = $false
                ErrorMessage = "Cannot extract file path from line 1"
            }
        }

        # Get staged file
        $stagedFileOutput = & git diff --cached --name-only 2>&1
        $stagedFileList = @($stagedFileOutput | Where-Object {
            $_ -and $_ -notmatch '^\s*$'
        })

        if ($stagedFileList.Count -eq 0) {
            return [PSCustomObject]@{
                IsValid = $false
                ErrorMessage = "No staged files found"
            }
        }

        $stagedFile = $stagedFileList[0].Trim()

        # Normalize paths (convert backslashes to forward slashes)
        $normalizedRef = $referencedPath -replace '\\', '/'
        $normalizedStaged = $stagedFile -replace '\\', '/'

        Write-DebugLog -Scope "HOOK-COMMITMSG" `
            -Message "Referenced: $normalizedRef, Staged: $normalizedStaged"

        # Case-insensitive comparison
        if ($normalizedRef -ne $normalizedStaged -and
            $normalizedRef.ToLower() -ne $normalizedStaged.ToLower()) {
            return [PSCustomObject]@{
                IsValid = $false
                ErrorMessage = "Line 1 file path ""$referencedPath"" " +
                    "does not match staged file ""$stagedFile"""
            }
        }

        return [PSCustomObject]@{
            IsValid = $true
            ErrorMessage = ''
        }
    }
    catch {
        Write-ErrorLog -Scope "HOOK-COMMITMSG" `
            -Message "Error validating path: $($_.Exception.Message)"
        throw
    }
}

function Test-BodyLineContent {
    <#
    .SYNOPSIS
        Validates each line has content after the prefix.

    .DESCRIPTION
        Checks if each body line contains non-whitespace content after
        removing the numbered prefix. Ensures meaningful information is
        provided for each required field.

    .PARAMETER BodyLines
        Array of body lines from the commit message.

    .OUTPUTS
        PSCustomObject with IsValid (bool) and ErrorMessage (string).

    .EXAMPLE
        $result = Test-BodyLineContent -BodyLines $lines
        Validates line content presence.

    .NOTES
        Returns IsValid=$true if all lines have content, otherwise
        IsValid=$false with line number.
    #>
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateCount(5, 5)]
        [string[]]$BodyLines
    )

    Write-DebugLog -Scope "HOOK-COMMITMSG" `
        -Message "Validating line content"

    $prefixes = @(
        '1. file:',
        '2. change:',
        '3. reason:',
        '4. impact:',
        '5. verify:'
    )

    for ($index = 0; $index -lt 5; $index++) {
        $line = $BodyLines[$index]
        $prefix = $prefixes[$index]
        $lineNumber = $index + 1

        # Remove prefix (case-insensitive) and trim
        $escapedPrefix = [regex]::Escape($prefix)
        $content = $line -replace "(?i)^$escapedPrefix", ''
        $content = $content.Trim()

        if ($content -eq '') {
            return [PSCustomObject]@{
                IsValid = $false
                ErrorMessage = "Line $lineNumber must have content " +
                    "after ""$prefix"""
            }
        }
    }

    return [PSCustomObject]@{
        IsValid = $true
        ErrorMessage = ''
    }
}

#endregion

#region Main Script Execution

Initialize-ScriptEnvironment
Assert-WindowsPlatform
Assert-PowerShellVersionStrict

try {
    Write-DebugLog -Scope "HOOK-COMMITMSG" `
        -Message "Starting commit message body validation"

    # Extract body lines from commit message
    $bodyLines = Get-CommitMessageBody -MessageFilePath $CommitMsgFile

    # Validate line count (exactly 5)
    $lineCountResult = Test-BodyLineCount -BodyLines $bodyLines
    if (-not $lineCountResult.IsValid) {
        Write-ErrorLog -Scope "HOOK-COMMITMSG" `
            -Message $lineCountResult.ErrorMessage
        exit 1
    }

    # Validate line prefixes
    $prefixResult = Test-BodyLinePrefixes -BodyLines $bodyLines
    if (-not $prefixResult.IsValid) {
        Write-ErrorLog -Scope "HOOK-COMMITMSG" `
            -Message $prefixResult.ErrorMessage
        exit 1
    }

    # Validate file path reference
    $filePathResult = Test-FilePathReference -BodyLines $bodyLines
    if (-not $filePathResult.IsValid) {
        Write-ErrorLog -Scope "HOOK-COMMITMSG" `
            -Message $filePathResult.ErrorMessage
        exit 1
    }

    # Validate line content
    $contentResult = Test-BodyLineContent -BodyLines $bodyLines
    if (-not $contentResult.IsValid) {
        Write-ErrorLog -Scope "HOOK-COMMITMSG" `
            -Message $contentResult.ErrorMessage
        exit 1
    }

    Write-InfoLog -Scope "HOOK-COMMITMSG" `
        -Message "Commit message body validation passed"

    exit 0
}
catch {
    Write-ExceptionLog -Scope "HOOK-COMMITMSG" `
        -Message "Unexpected error: $($_.Exception.Message)"

    Write-DebugLog -Scope "HOOK-COMMITMSG" `
        -Message "Stack Trace: $($_.ScriptStackTrace)"

    exit 1
}

#endregion
