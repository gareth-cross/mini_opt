$build_config = $args[0]
$build_cmd = "MSBuild.exe `"mini_opt.sln`" -nologo -verbosity:m -property:Configuration=$build_config"
Write-host "Invoking: $build_cmd"
Invoke-Expression $build_cmd
