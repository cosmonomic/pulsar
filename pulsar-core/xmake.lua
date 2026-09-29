set_project("pulsar-core")
set_xmakever("3.0.0")

add_rules("mode.release", "mode.debug")
set_defaultmode("debug")

option("torch_cmake_prefix", {default = "", description = "Torch's CMake prefix path, used to configure libtorch."})
option("prof", {default = false, description = "Compile the PULSAR_PROF phase timers into the engine."})
option("tests", {default = true, description = "Build the Catch2 test binaries."})

add_requires("cmake::Torch", {system = true, configs = {
    envs = {CMAKE_PREFIX_PATH = get_config("torch_cmake_prefix")},
    presets = {CMAKE_LINK_DEPENDS_USE_LINKER = false},
}})

target("pulsar_core")
    set_kind("static")
    set_languages("c++23")
    add_cugencodes("native")
    set_values("cuda.rdc", false)
    add_cxflags("-fPIC")
    add_cuflags("-std=c++23", "-Xcompiler=-fPIC", "--expt-relaxed-constexpr", "--expt-extended-lambda")
    add_files("src/**.cu", "src/**.cpp")
    add_includedirs("include", {public = true})
    add_headerfiles("include/(**)")
    add_packages("cmake::Torch", {public = true})
    if has_config("prof") then
        add_defines("PULSAR_PROF", {public = true})
    end
target_end()

if has_config("tests") then
    add_requires("catch2 3.5.4")

    for _, file in ipairs(os.files(path.join(os.scriptdir(), "tests/**/test_*.cpp"))) do
        target(path.basename(file))
            set_kind("binary")
            set_group("tests")
            set_languages("c++23")
            add_files(file)
            add_deps("pulsar_core")
            add_packages("catch2")
            add_tests("default")
        target_end()
    end
end
