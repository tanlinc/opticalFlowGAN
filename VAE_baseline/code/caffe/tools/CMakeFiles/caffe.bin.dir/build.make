# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe

# Include any dependencies generated for this target.
include tools/CMakeFiles/caffe.bin.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/caffe.bin.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/caffe.bin.dir/flags.make

tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o: tools/CMakeFiles/caffe.bin.dir/flags.make
tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o: tools/caffe.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o"
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffe.bin.dir/caffe.cpp.o -c /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools/caffe.cpp

tools/CMakeFiles/caffe.bin.dir/caffe.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffe.bin.dir/caffe.cpp.i"
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools/caffe.cpp > CMakeFiles/caffe.bin.dir/caffe.cpp.i

tools/CMakeFiles/caffe.bin.dir/caffe.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffe.bin.dir/caffe.cpp.s"
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools/caffe.cpp -o CMakeFiles/caffe.bin.dir/caffe.cpp.s

tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o.requires:

.PHONY : tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o.requires

tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o.provides: tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/caffe.bin.dir/build.make tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o.provides.build
.PHONY : tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o.provides

tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o.provides.build: tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o


# Object files for target caffe.bin
caffe_bin_OBJECTS = \
"CMakeFiles/caffe.bin.dir/caffe.cpp.o"

# External object files for target caffe.bin
caffe_bin_EXTERNAL_OBJECTS =

tools/caffe: tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o
tools/caffe: tools/CMakeFiles/caffe.bin.dir/build.make
tools/caffe: lib/libcaffe.so.1.0.0-rc3
tools/caffe: lib/libproto.a
tools/caffe: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libglog.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/caffe: /usr/local/lib/libprotobuf.so
tools/caffe: /home/linkermann/anaconda2/lib/libhdf5_cpp.so
tools/caffe: /home/linkermann/anaconda2/lib/libhdf5.so
tools/caffe: /usr/lib/x86_64-linux-gnu/librt.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libglog.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/caffe: /usr/local/lib/libprotobuf.so
tools/caffe: /home/linkermann/anaconda2/lib/libhdf5_cpp.so
tools/caffe: /home/linkermann/anaconda2/lib/libhdf5.so
tools/caffe: /usr/lib/x86_64-linux-gnu/librt.so
tools/caffe: /home/linkermann/anaconda2/lib/libz.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libdl.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libm.so
tools/caffe: /home/linkermann/anaconda2/lib/libhdf5_hl_cpp.so
tools/caffe: /home/linkermann/anaconda2/lib/libhdf5_hl.so
tools/caffe: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libsnappy.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
tools/caffe: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
tools/caffe: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
tools/caffe: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
tools/caffe: /usr/lib/x86_64-linux-gnu/liblapack.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libcblas.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libatlas.so
tools/caffe: /home/linkermann/anaconda2/lib/libpython2.7.so
tools/caffe: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/caffe: tools/CMakeFiles/caffe.bin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable caffe"
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffe.bin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/caffe.bin.dir/build: tools/caffe

.PHONY : tools/CMakeFiles/caffe.bin.dir/build

tools/CMakeFiles/caffe.bin.dir/requires: tools/CMakeFiles/caffe.bin.dir/caffe.cpp.o.requires

.PHONY : tools/CMakeFiles/caffe.bin.dir/requires

tools/CMakeFiles/caffe.bin.dir/clean:
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && $(CMAKE_COMMAND) -P CMakeFiles/caffe.bin.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/caffe.bin.dir/clean

tools/CMakeFiles/caffe.bin.dir/depend:
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools/CMakeFiles/caffe.bin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/caffe.bin.dir/depend

