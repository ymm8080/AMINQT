# -*- coding: utf-8 -*-
$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== Step 1: Empty Recycle Bin on D: ==="
Clear-RecycleBin -DriveLetter D -Force -ErrorAction SilentlyContinue
Write-Host "Recycle bin cleared."
Write-Host ""

Write-Host "=== Step 2: Delete node_modules ==="
$nmDirs = Get-ChildItem -Path 'D:\' -Directory -Recurse -Depth 5 -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq 'node_modules' }
$nmCount = 0
foreach ($d in $nmDirs) {
    Write-Host "  Deleting: $($d.FullName)"
    Remove-Item -Path $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
    $nmCount++
}
Write-Host "Deleted $nmCount node_modules directories."
Write-Host ""

Write-Host "=== Step 3: Delete __pycache__ ==="
$pycDirs = Get-ChildItem -Path 'D:\' -Directory -Recurse -Depth 5 -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq '__pycache__' }
$pycCount = 0
foreach ($d in $pycDirs) {
    Remove-Item -Path $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
    $pycCount++
}
Write-Host "Deleted $pycCount __pycache__ directories."
Write-Host ""

Write-Host "=== Step 4: Delete .pytest_cache ==="
$ptDirs = Get-ChildItem -Path 'D:\' -Directory -Recurse -Depth 5 -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq '.pytest_cache' }
$ptCount = 0
foreach ($d in $ptDirs) {
    Remove-Item -Path $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
    $ptCount++
}
Write-Host "Deleted $ptCount .pytest_cache directories."
Write-Host ""

Write-Host "=== Step 5: Delete .ruff_cache ==="
$ruffDirs = Get-ChildItem -Path 'D:\' -Directory -Recurse -Depth 5 -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq '.ruff_cache' }
$ruffCount = 0
foreach ($d in $ruffDirs) {
    Remove-Item -Path $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
    $ruffCount++
}
Write-Host "Deleted $ruffCount .ruff_cache directories."
Write-Host ""

Write-Host "=== Step 6: Delete cache directories ==="
$cacheDirs = Get-ChildItem -Path 'D:\' -Directory -Recurse -Depth 5 -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq 'cache' }
$cacheCount = 0
foreach ($d in $cacheDirs) {
    Write-Host "  Deleting: $($d.FullName)"
    Remove-Item -Path $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
    $cacheCount++
}
Write-Host "Deleted $cacheCount cache directories."
Write-Host ""

Write-Host "=== Final D: Drive Space ==="
$d = Get-PSDrive D
$usedGB = [math]::Round($d.Used / 1GB, 2)
$freeGB = [math]::Round($d.Free / 1GB, 2)
Write-Host "Used: $usedGB GB | Free: $freeGB GB"
