# Project Overview
This project is a CUDA and libtorch-based template, intended to serve as a starting point for high performance computing and machine learning projects on NVIDIA GPUs. It uses Google Test as a testing framework to test against libtorch.

The project uses cmake for the build process, and the cmake file is set up in a way that it can be easily adapted to different projects. It includes configurations for both debug and release modes.

## libtorch

Download the libtorch library using the following command:

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
```

This will download a zip file named `libtorch-shared-with-deps-latest.zip`. To extract this zip file, use the command:

```bash
unzip libtorch-shared-with-deps-latest.zip -d external/
rm libtorch-shared-with-deps-latest.zip
```

## Building the Project

To build the project, follow these steps:
1. Create a new directory named `build` and navigate into it:

    ```bash
    mkdir build && cd build
    ```

2. Run the CMake configuration:

    ```bash
    cmake -DCMAKE_BUILD_TYPE=Release ..
    ```
	If you have problems to configure the build, you might look up the graphics card architecture you are using.
	Then replace CUDA_ARCHITECTURE 89 with the number of your architecture.

3. Finally, compile the project:

    ```bash
    make -j$(nproc)
    cd ..
    ```

This will create an executable named `cuda_libtorch_gtest_template` and `cuda_kernel_tests` in the `build` directory.

## CMake Build Process
The CMake build process is structured as follows:

*    It first sets up the project root directory and defines the absolute path to libtorch. The CMAKE_PREFIX_PATH is set to the libtorch directory.

*    The compile definition _GLIBCXX_USE_CXX11_ABI=0 is added. This is necessary to avoid linker errors with libtorch.

*    The project is then defined with CUDA and CXX languages.

*    It determines the number of cores in the system and reserves two cores for system operations, using the rest for the build process.

*    The header files are then defined, followed by the source files. The executable is then added.

*    It specifies that the CUDA and CXX standards are 17 and these standards are required.

*    It sets up libtorch, ensuring that the Torch package is found and linked correctly.

*    It includes conditional compilation options for Debug and Release modes.

*    CUDA Toolkit and TBB (Threading Building Blocks) packages are found and linked.

*    It includes a check for CUDA version, ensuring that the version is 12.0 or higher.

*    It then sets up Google Test manually, making sure to avoid treating all warnings as errors.

*    It enables CTest and adds a test executable.

*    Finally, it specifies the properties for the test executable and the libraries it links to.

This CMake setup ensures a flexible and robust build process for CUDA and libtorch projects, making it easier to start new projects and ensuring that all dependencies are correctly included and linked. It also provides a structure for unit testing with Google Test, making sure that the project can be tested thoroughly.

Please remember to select the correct architecture for your GPU in the CMakeLists.txt file. The current setting is CUDA_ARCHITECTURES 89 which corresponds to the architecture of RTX 4090. For other GPUs, please replace 89 with the correct architecture number.


