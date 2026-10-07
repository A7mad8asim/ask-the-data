# Fix for Ollama on Windows: a model downloads fine but is missing from `ollama list`,
# and %LOCALAPPDATA%\Ollama\server.log shows "bad manifest ... untrusted mount point".
#
# Newer Ollama versions keep a second ("v2") manifest for each model as a symbolic link.
# Windows' redirection protection can stop the Ollama server from following that link.
# This script replaces each such link with a regular copy of the file it points to.
# The content is identical, so nothing else changes. Re-pulling the model would recreate the link.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\fix-ollama-manifests.ps1

$models = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } else { Join-Path $env:USERPROFILE ".ollama\models" }
$v2 = Join-Path $models "manifests-v2"
if (-not (Test-Path $v2)) {
    Write-Host "Nothing to fix: $v2 does not exist."
    return
}

$fixed = 0
Get-ChildItem -Path $v2 -Recurse -File -Force | Where-Object { $_.LinkType -eq "SymbolicLink" } | ForEach-Object {
    $target = $_.Target | Select-Object -First 1
    if (-not [IO.Path]::IsPathRooted($target)) { $target = Join-Path $_.DirectoryName $target }
    $target = [IO.Path]::GetFullPath($target)
    if (-not (Test-Path $target)) {
        Write-Warning "Skipped $($_.FullName): its target $target is missing (pull the model again)."
        return
    }
    $bytes = [IO.File]::ReadAllBytes($target)
    $_.Delete()  # removes the link only, never the file it points to
    [IO.File]::WriteAllBytes($_.FullName, $bytes)
    Write-Host "Fixed $(($_.FullName -split '\\manifests-v2\\', 2)[1])"
    $fixed++
}
Write-Host "$fixed manifest link(s) replaced. Check with:  ollama list"
