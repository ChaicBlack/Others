# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.27.7/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.27.7/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zihaoshen/projects/Others

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zihaoshen/projects/Others/build

# Include any dependencies generated for this target.
include CMakeFiles/others.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/others.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/others.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/others.dir/flags.make

CMakeFiles/others.dir/src/list2_8.cpp.o: CMakeFiles/others.dir/flags.make
CMakeFiles/others.dir/src/list2_8.cpp.o: /Users/zihaoshen/projects/Others/src/list2_8.cpp
CMakeFiles/others.dir/src/list2_8.cpp.o: CMakeFiles/others.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/zihaoshen/projects/Others/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/others.dir/src/list2_8.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/others.dir/src/list2_8.cpp.o -MF CMakeFiles/others.dir/src/list2_8.cpp.o.d -o CMakeFiles/others.dir/src/list2_8.cpp.o -c /Users/zihaoshen/projects/Others/src/list2_8.cpp

CMakeFiles/others.dir/src/list2_8.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/others.dir/src/list2_8.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zihaoshen/projects/Others/src/list2_8.cpp > CMakeFiles/others.dir/src/list2_8.cpp.i

CMakeFiles/others.dir/src/list2_8.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/others.dir/src/list2_8.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zihaoshen/projects/Others/src/list2_8.cpp -o CMakeFiles/others.dir/src/list2_8.cpp.s

# Object files for target others
others_OBJECTS = \
"CMakeFiles/others.dir/src/list2_8.cpp.o"

# External object files for target others
others_EXTERNAL_OBJECTS =

others: CMakeFiles/others.dir/src/list2_8.cpp.o
others: CMakeFiles/others.dir/build.make
others: CMakeFiles/others.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/zihaoshen/projects/Others/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable others"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/others.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/others.dir/build: others
.PHONY : CMakeFiles/others.dir/build

CMakeFiles/others.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/others.dir/cmake_clean.cmake
.PHONY : CMakeFiles/others.dir/clean

CMakeFiles/others.dir/depend:
	cd /Users/zihaoshen/projects/Others/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zihaoshen/projects/Others /Users/zihaoshen/projects/Others /Users/zihaoshen/projects/Others/build /Users/zihaoshen/projects/Others/build /Users/zihaoshen/projects/Others/build/CMakeFiles/others.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/others.dir/depend

