<#
Registers ClaudeYe for ALL Claude Code projects (user scope):
  1. MCP server  -> claude mcp add -s user claudeye -- <this repo>\.venv\Scripts\python.exe -m claudeye
  2. Skill       -> %USERPROFILE%\.claude\skills\claudeye-vision\
  3. Subagent    -> %USERPROFILE%\.claude\agents\vision-analyst.md

Dry run by default. Apply with:   powershell -ExecutionPolicy Bypass -File scripts\install_claude.ps1 -Apply
Options: -AllowCloud (lets ask_vision/ask_video use Gemini; also set GEMINI_API_KEY in your environment)
Undo:    claude mcp remove -s user claudeye; delete the copied skill folder and agent file.
#>
param([switch]$Apply, [switch]$AllowCloud)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No venv at $py - run 'uv sync --extra ...' in $root first (see README)." }

$claudeHome = Join-Path $env:USERPROFILE ".claude"
$skillSrc = Join-Path $root ".claude\skills\claudeye-vision"
$skillDst = Join-Path $claudeHome "skills\claudeye-vision"
$agentSrc = Join-Path $root ".claude\agents\vision-analyst.md"
$agentDst = Join-Path $claudeHome "agents\vision-analyst.md"
$cloud = if ($AllowCloud) { "1" } else { "0" }
$mcpArgs = @("mcp", "add", "-s", "user", "claudeye", "-e", "CLAUDEYE_MAX_EDGE=1568", "-e", "CLAUDEYE_ALLOW_CLOUD=$cloud", "--", $py, "-m", "claudeye")

Write-Host "Planned actions:"
Write-Host "  claude $($mcpArgs -join ' ')"
Write-Host "  copy $skillSrc -> $skillDst"
Write-Host "  copy $agentSrc -> $agentDst"
if (-not $Apply) { Write-Host "`nDry run only. Re-run with -Apply to perform these steps."; exit 0 }

& claude mcp remove -s user claudeye 2>$null | Out-Null
& claude @mcpArgs
New-Item -ItemType Directory -Force (Split-Path $skillDst) | Out-Null
New-Item -ItemType Directory -Force (Split-Path $agentDst) | Out-Null
Copy-Item -Recurse -Force $skillSrc (Split-Path $skillDst)
Copy-Item -Force $agentSrc $agentDst
Write-Host "`nDone. Restart Claude Code, then run /mcp to confirm 'claudeye' is connected."
