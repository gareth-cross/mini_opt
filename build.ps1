param (
    [Parameter(Mandatory)]$install_prefix, 
    [string]$eigen_directory, 
    [switch]$clean,
    [string]$build_type='Release',
    [switch]$install_third_party
)

# Turn this into a VS dev shell:
# See https://intellitect.com/enter-vsdevshell-powershell/
$installPath = &"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -version 16.0 -property installationpath
Import-Module (Join-Path $installPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $installPath -SkipAutomaticLocation

# Determine where the script lives
$script_path = $MyInvocation.MyCommand.Path
$script_directory = Split-Path $script_path
Write-host "The project directory is $script_directory"

$build_dir = (Join-Path $script_directory "build")
if ($clean) {
    Write-Host "Will run a clean build."
    if (Test-Path $build_dir) {
        # Nuke the build directory
        Remove-Item -Recurse $build_dir
    }
}

# Make the build-directory
if (!(Test-Path $build_dir)) {
    Write-Host "Creating: $build_dir"
    $null = New-Item -ItemType Directory -Force -Path $build_dir
}

# Enter the build-directory
Push-Location
Set-Location $build_dir

# Run cmake
$external_depends_args = "-DCMAKE_BUILD_TYPE=$build_type -DCMAKE_INSTALL_PREFIX=`"$install_prefix`""
if ($eigen_directory) {
    $external_depends_args = $external_depends_args + " -DEIGEN_DIRECTORY=`"$eigen_directory`""
}
if ($install_third_party) {
    $external_depends_args = $external_depends_args + ' -DINSTALL_LIBFMT=1 -DINSTALL_EIGEN=1'
}
$cmake_cmd = "cmake `"$script_directory`" $external_depends_args"

Write-Host "Invoking: $cmake_cmd"
try {
    Invoke-Expression $cmake_cmd
} catch {
    Write-Host "Failed to run cmake!"
    return;
}

# Now invoke the powershell makefile:
& "$build_dir\make.ps1" -install

Pop-Location
