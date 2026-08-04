param(
  [string]$DbPath = "C:\Users\Lalo\Documents\Code\jarvis\.codegraph\codegraph.db",
  [string]$OutDir = "C:\Users\Lalo\Documents\Code\jarvis\codegraph"
)

Write-Host "=== CodeGraph Viz Sync ===" -ForegroundColor Cyan
Write-Host "DB: $DbPath"
Write-Host "Out: $OutDir"
Write-Host ""

# Check DB exists
if (-not (Test-Path $DbPath)) {
  Write-Host "ERROR: Database not found at $DbPath" -ForegroundColor Red
  exit 1
}

# Export nodes
Write-Host "Exporting nodes..." -NoNewline
$nodesJson = sqlite3 -json $DbPath @"
SELECT id, name, kind, file_path, qualified_name
FROM nodes
WHERE kind IN ('class','interface','method','enum','route')
ORDER BY kind, name;
"@
if (-not $nodesJson) {
  Write-Host " FAILED (empty)" -ForegroundColor Red
  exit 1
}
"const NODES_DATA = $nodesJson;" | Set-Content -Path "$OutDir\nodes.js" -Encoding UTF8
$nodeCount = ($nodesJson | ConvertFrom-Json).Count
Write-Host " $nodeCount nodes" -ForegroundColor Green

# Export edges (real + synthetic transitive for nodes bridged via field/enum_member)
Write-Host "Exporting edges..." -NoNewline
$edgesJson = sqlite3 -json $DbPath @"
SELECT source, target, kind FROM edges
WHERE kind IN ('calls','extends','implements','references','instantiates','contains')
UNION
SELECT DISTINCT e1.source AS source, e2.target AS target, 'references' AS kind
FROM edges e1
JOIN edges e2 ON e1.target = e2.source
WHERE e1.kind = 'contains' AND e2.kind = 'references'
  AND e1.source IN (SELECT id FROM nodes WHERE kind IN ('class','interface','method','enum','route'))
  AND e2.target IN (SELECT id FROM nodes WHERE kind IN ('class','interface','method','enum','route'))
UNION
SELECT DISTINCT e1.source AS source, e2.source AS target, 'references' AS kind
FROM edges e1
JOIN edges e2 ON e1.target = e2.target
WHERE e1.kind = 'references' AND e2.kind = 'contains'
  AND e1.source IN (SELECT id FROM nodes WHERE kind IN ('class','interface','method','enum','route'))
  AND e2.source IN (SELECT id FROM nodes WHERE kind IN ('class','interface','method','enum','route'));
"@
if (-not $edgesJson) {
  Write-Host " FAILED (empty)" -ForegroundColor Red
  exit 1
}
"const EDGES_DATA = $edgesJson;" | Set-Content -Path "$OutDir\edges.js" -Encoding UTF8
$edgeCount = ($edgesJson | ConvertFrom-Json).Count
Write-Host " $edgeCount edges" -ForegroundColor Green

# File sizes
$nodesSize = (Get-Item "$OutDir\nodes.js").Length
$edgesSize = (Get-Item "$OutDir\edges.js").Length
Write-Host ""
Write-Host "Done:" -ForegroundColor Cyan
Write-Host "  nodes.js  ($([math]::Round($nodesSize/1KB)) KB)"
Write-Host "  edges.js  ($([math]::Round($edgesSize/1KB)) KB)"
Write-Host ""
Write-Host "Open $OutDir\index.html in your browser to view." -ForegroundColor Yellow
