includes(os.isdir(path.join(os.scriptdir(), "pulsar-core")) and "pulsar-core" or "../pulsar-core")

set_project("pulsar")

option("torch_cmake_prefix")
    after_check(function (option)
        if option:value() == "" then
            option:set_value(os.iorunv("python", {"-c", "import torch.utils; print(torch.utils.cmake_prefix_path)"}):trim())
        end
    end)
option_end()

target("pulsar")
    set_kind("shared")
    set_languages("c++23")
    set_basename("pulsar_core")
    set_prefixname("")
    add_files("ext/*.cpp")
    add_deps("pulsar_core")
    add_packages("cmake::Torch")
    on_load(function (target)
        target:add("shflags", target:pkg("cmake::Torch"):get("ldflags"))
    end)
