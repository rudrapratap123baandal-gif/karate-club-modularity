@echo off
cd /d "c:\Users\rudra\Documents\antigravity\karate-club-modularity"
echo Pushing DSC212 Karate Club Modularity repository to GitHub...
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================================
    echo SUCCESS! Repository published to:
    echo https://github.com/rudrapratap123baandal-gif/karate-club-modularity
    echo ========================================================
) else (
    echo.
    echo If prompted to authenticate, please complete sign-in in the popup window.
)
pause
