param (
    [string]$build_type='Release',
    [string]$verbosity='m',
    [switch]$install
)
$build_cmd = "MSBuild.exe ALL_BUILD.vcxproj -nologo -verbosity:$verbosity -property:Configuration=$build_type"
Write-host "Invoking: $build_cmd"
Invoke-Expression $build_cmd

if ($install) {
    # build succeeded, run the install target
    $install_cmd = "MSBuild.exe INSTALL.vcxproj -verbosity:$verbosity -property:Configuration=$build_type"
    Write-host "Invoking: $install_cmd"
    Invoke-Expression $install_cmd
}
