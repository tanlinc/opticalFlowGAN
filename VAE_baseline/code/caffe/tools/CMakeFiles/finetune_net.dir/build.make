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
include tools/CMakeFiles/finetune_net.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/finetune_net.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/finetune_net.dir/flags.make

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o: tools/CMakeFiles/finetune_net.dir/flags.make
tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o: tools/finetune_net.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o"
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/finetune_net.dir/finetune_net.cpp.o -c /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools/finetune_net.cpp

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/finetune_net.dir/finetune_net.cpp.i"
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools/finetune_net.cpp > CMakeFiles/finetune_net.dir/finetune_net.cpp.i

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/finetune_net.dir/finetune_net.cpp.s"
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools/finetune_net.cpp -o CMakeFiles/finetune_net.dir/finetune_net.cpp.s

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.requires:

.PHONY : tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.requires

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.provides: tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/finetune_net.dir/build.make tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.provides.build
.PHONY : tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.provides

tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.provides.build: tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o


# Object files for target finetune_net
finetune_net_OBJECTS = \
"CMakeFiles/finetune_net.dir/finetune_net.cpp.o"

# External object files for target finetune_net
finetune_net_EXTERNAL_OBJECTS =

tools/finetune_net: tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o
tools/finetune_net: tools/CMakeFiles/finetune_net.dir/build.make
tools/finetune_net: lib/libcaffe.so.1.0.0-rc3
tools/finetune_net: lib/libproto.a
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libglog.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/finetune_net: /usr/local/lib/libprotobuf.so
tools/finetune_net: /home/linkermann/anaconda2/lib/libhdf5_cpp.so
tools/finetune_net: /home/linkermann/anaconda2/lib/libhdf5.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/librt.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libglog.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/finetune_net: /usr/local/lib/libprotobuf.so
tools/finetune_net: /home/linkermann/anaconda2/lib/libhdf5_cpp.so
tools/finetune_net: /home/linkermann/anaconda2/lib/libhdf5.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/librt.so
tools/finetune_net: /home/linkermann/anaconda2/lib/libz.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libdl.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libm.so
tools/finetune_net: /home/linkermann/anaconda2/lib/libhdf5_hl_cpp.so
tools/finetune_net: /home/linkermann/anaconda2/lib/libhdf5_hl.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libsnappy.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
tools/finetune_net: /usr/lib/x86_64-linux-gnu/liblapack.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libcblas.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libatlas.so
tools/finetune_net: /home/linkermann/anaconda2/lib/libpython2.7.so
tools/finetune_net: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/finetune_net: tools/CMakeFiles/finetune_net.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable finetune_net"
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/finetune_net.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/finetune_net.dir/build: tools/finetune_net

.PHONY : tools/CMakeFiles/finetune_net.dir/build

tools/CMakeFiles/finetune_net.dir/requires: tools/CMakeFiles/finetune_net.dir/finetune_net.cpp.o.requires

.PHONY : tools/CMakeFiles/finetune_net.dir/requires

tools/CMakeFiles/finetune_net.dir/clean:
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools && $(CMAKE_COMMAND) -P CMakeFiles/finetune_net.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/finetune_net.dir/clean

tools/CMakeFiles/finetune_net.dir/depend:
	cd /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools /home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/VAE_baseline/code/caffe/tools/CMakeFiles/finetune_net.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/finetune_net.dir/depend

