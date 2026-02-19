@echo off
setlocal EnableDelayedExpansion

:: Change to script directory (and save original)
pushd "%~dp0"

:: Defaults
set CONFIG=Release
set RUN_TESTS=false
set CLEAN=false

:: Parse Arguments
:parse_loop
if "%~1"=="" goto check_env
if "%~1"=="--debug" (
    set CONFIG=Debug
) else if "%~1"=="--test" (
    set RUN_TESTS=true
) else if "%~1"=="--clean" (
    set CLEAN=true
) else (
    echo Unknown parameter: %~1
    exit /b 1
)
shift
goto parse_loop

:check_env
:: Check if we are already in a VS Command Prompt (cl.exe exists)
where cl.exe >nul 2>nul
if %errorlevel% equ 0 goto check_vcpkg

echo Initializing Visual Studio Environment...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Error: vswhere.exe not found. Is Visual Studio installed?
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
)

if not defined VS_PATH (
    echo Error: Visual Studio with C++ tools not found.
    exit /b 1
)

call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo Error: Failed to initialize VS environment.
    exit /b 1
)
echo VS Environment Initialized.

:check_vcpkg
:: 1. Check Global VCPKG (Disabled: Force Local to ensure compatibility)
:: if defined VCPKG_ROOT if exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
::    echo Found Global VCPKG at "%VCPKG_ROOT%"
::    set "VCPKG_CMAKE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
::    goto build_start
:: )

:: 2. Check Local VCPKG
if not exist "vcpkg" (
    echo Cloning vcpkg locally...
    git clone https://github.com/microsoft/vcpkg.git
    if errorlevel 1 exit /b 1
)

if not exist "vcpkg\vcpkg.exe" (
    echo Bootstrapping vcpkg...
    call "vcpkg\bootstrap-vcpkg.bat"
    if errorlevel 1 exit /b 1
)

echo Using Local VCPKG.
set "VCPKG_ROOT=%~dp0vcpkg"
set "VCPKG_CMAKE=vcpkg/scripts/buildsystems/vcpkg.cmake"

:: Generate vcpkg-configuration.json with current baseline
for /f "tokens=*" %%g in ('git -C vcpkg rev-parse HEAD') do (set VCPKG_COMMIT=%%g)
echo Configuring vcpkg baseline to %VCPKG_COMMIT%...
(
    echo {
    echo   "default-registry": {
    echo     "kind": "git",
    echo     "repository": "https://github.com/microsoft/vcpkg",
    echo     "baseline": "%VCPKG_COMMIT%"
    echo   }
    echo }
) > vcpkg-configuration.json

:build_start
echo --- POPGP Engine Build (%CONFIG%) ---

:: 1. Clean
if "%CLEAN%"=="true" (
    if exist build (
        echo Cleaning build directory...
        rmdir /s /q build
    )
)

:: 2. Configure
if not exist build\CMakeCache.txt (
    echo Configuring CMake...
    
    :: Use Visual Studio Generator (Safest on Windows)
    :: Pass VCPKG toolchain explicitly with quotes
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=!CONFIG! "-DCMAKE_TOOLCHAIN_FILE=!VCPKG_CMAKE!"
    if errorlevel 1 exit /b %errorlevel%
)

:: 3. Build
echo Building...
cmake --build build --config !CONFIG!
if errorlevel 1 exit /b %errorlevel%

:: 4. Tests
if "%RUN_TESTS%"=="true" (
    echo Running Tests...
    cd build
    ctest -C !CONFIG! --output-on-failure
    cd ..
)

echo Build Complete!
popd
endlocal
