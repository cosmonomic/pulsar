set_project("pulsar-core")
set_xmakever("3.0.0")

add_rules("mode.release", "mode.debug")
set_defaultmode("debug")
set_policy("build.optimization.lto", true)
add_ldflags("-flto=auto")
add_shflags("-flto=auto")

option("torch_cmake_prefix", {default = "", description = "Torch's CMake prefix path, used to configure libtorch."})
option("prof", {default = false, description = "Compile the PULSAR_PROF phase timers into the engine."})
option("tests", {default = true, description = "Build the Catch2 test binaries."})
option("digest", {default = "sha256", values = {"sha256", "blake3"}, description = "Hash function behind page digests."})

if is_config("digest", "blake3") then
    add_requires("blake3 1.8.7")
else
    add_requires("openssl3", {system = true})
end
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
    add_files("src/**.cu", "src/**.cpp|digest_*.cpp")
    add_files("src/digest_$(digest).cpp")
    add_includedirs("inc", {public = true})
    add_headerfiles("inc/(**)")
    add_packages("cmake::Torch", {public = true})
    if is_config("digest", "blake3") then
        add_packages("blake3")
    else
        add_packages("openssl3")
    end
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
