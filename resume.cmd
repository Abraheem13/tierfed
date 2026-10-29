@echo off
cd /d C:\Users\athawerani\Documents\Abraheem_work\paper3\temp\tierfed-import
powershell.exe -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1 -Stages "2 3 4 5 6 7" -Jobs 2 > reproduce.log 2> reproduce.err
