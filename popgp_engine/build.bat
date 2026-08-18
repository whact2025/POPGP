@echo off
setlocal EnableDelayedExpansion

:: Change to script directory (and save original)
pushd "%~dp0"

:: Defaults
set CONFIG=Release
set RUN_TESTS=false
set CLEAN=false
set CUDA_ARCH=native
set VCPKG_COMMIT=e5a1490e1409d175932ef6014519e9ae149ddb7c

:: Parse Arguments
:parse_loop
if "%~1"=="" goto check_env
if "%~1"=="--debug" (
    set CONFIG=Debug
) else if "%~1"=="--test" (
    set RUN_TESTS=true
) else if "%~1"=="--clean" (
    set CLEAN=true
) else if "%~1"=="--cuda-arch" (
    goto parse_cuda_arch
) else (
    echo Unknown parameter: %~1
    exit /b 1
)
shift
goto parse_loop

:parse_cuda_arch
shift
if "%~1"=="" (
    echo Error: --cuda-arch requires a CMake CUDA architecture value.
    exit /b 1
)
set CUDA_ARCH=%~1
shift
goto parse_loop

:check_env
cmake -DPOPGP_CUDA_ARCHITECTURE=!CUDA_ARCH! -P cmake/ValidateCudaArchitecture.cmake
if errorlevel 1 exit /b 1

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

git -C vcpkg cat-file -e "%VCPKG_COMMIT%^{commit}" >nul 2>nul
if errorlevel 1 (
    echo Fetching pinned vcpkg commit %VCPKG_COMMIT%...
    git -C vcpkg fetch --depth 1 origin %VCPKG_COMMIT%
    if errorlevel 1 exit /b 1
)

git -C vcpkg checkout --detach %VCPKG_COMMIT% >nul
if errorlevel 1 exit /b 1

echo Bootstrapping pinned vcpkg...
call "vcpkg\bootstrap-vcpkg.bat" -disableMetrics
if errorlevel 1 exit /b 1

echo Using Local VCPKG.
set "VCPKG_ROOT=%~dp0vcpkg"
set "VCPKG_CMAKE=vcpkg/scripts/buildsystems/vcpkg.cmake"

:build_start
echo --- POPGP Engine Build (%CONFIG%, CUDA architecture %CUDA_ARCH%) ---

:: 1. Clean
if "%CLEAN%"=="true" (
    if exist build (
        echo Cleaning build directory...
        rmdir /s /q build
    )
)

:: 2. Configure. Always rerun so requested architecture/configuration changes
:: cannot be silently masked by an existing CMake cache.
echo Configuring CMake...

:: Ninja uses the MSVC environment initialized above and does not require
:: version-specific CUDA Visual Studio integration.
where ninja.exe >nul 2>nul
if errorlevel 1 (
    echo Error: ninja.exe not found. Install Ninja or add it to PATH.
    exit /b 1
)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=!CONFIG! -DCMAKE_CUDA_ARCHITECTURES=!CUDA_ARCH! -DPOPGP_REQUIRE_VISIBLE_CUDA_ARCH=ON "-DCMAKE_TOOLCHAIN_FILE=!VCPKG_CMAKE!"
if errorlevel 1 exit /b 1

:: 3. Build
echo Building...
cmake --build build --config !CONFIG!
if errorlevel 1 exit /b 1

:: 4. Tests
if "%RUN_TESTS%"=="true" (
    echo Running Tests...
    cmake "-DPOPGP_BUILD_DIR=%CD%/build" -DPOPGP_CONFIG=!CONFIG! -P cmake/VerifyCTestCount.cmake
    if errorlevel 1 exit /b 1
    ctest --test-dir build -C !CONFIG! --output-on-failure --no-tests=error
    if errorlevel 1 exit /b 1
)

echo Build Complete!
popd
endlocal
