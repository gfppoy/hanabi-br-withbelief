# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

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
CMAKE_COMMAND = /opt/conda/bin/cmake

# The command to remove a file.
RM = /opt/conda/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /sad

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /sad/build

# Include any dependencies generated for this target.
include hanabi-learning-environment/CMakeFiles/game_example.dir/depend.make

# Include the progress variables for this target.
include hanabi-learning-environment/CMakeFiles/game_example.dir/progress.make

# Include the compile flags for this target's objects.
include hanabi-learning-environment/CMakeFiles/game_example.dir/flags.make

hanabi-learning-environment/CMakeFiles/game_example.dir/game_example.cc.o: hanabi-learning-environment/CMakeFiles/game_example.dir/flags.make
hanabi-learning-environment/CMakeFiles/game_example.dir/game_example.cc.o: ../hanabi-learning-environment/game_example.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/sad/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object hanabi-learning-environment/CMakeFiles/game_example.dir/game_example.cc.o"
	cd /sad/build/hanabi-learning-environment && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/game_example.dir/game_example.cc.o -c /sad/hanabi-learning-environment/game_example.cc

hanabi-learning-environment/CMakeFiles/game_example.dir/game_example.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/game_example.dir/game_example.cc.i"
	cd /sad/build/hanabi-learning-environment && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /sad/hanabi-learning-environment/game_example.cc > CMakeFiles/game_example.dir/game_example.cc.i

hanabi-learning-environment/CMakeFiles/game_example.dir/game_example.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/game_example.dir/game_example.cc.s"
	cd /sad/build/hanabi-learning-environment && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /sad/hanabi-learning-environment/game_example.cc -o CMakeFiles/game_example.dir/game_example.cc.s

# Object files for target game_example
game_example_OBJECTS = \
"CMakeFiles/game_example.dir/game_example.cc.o"

# External object files for target game_example
game_example_EXTERNAL_OBJECTS =

hanabi-learning-environment/game_example: hanabi-learning-environment/CMakeFiles/game_example.dir/game_example.cc.o
hanabi-learning-environment/game_example: hanabi-learning-environment/CMakeFiles/game_example.dir/build.make
hanabi-learning-environment/game_example: hanabi-learning-environment/hanabi_lib/libhanabi.a
hanabi-learning-environment/game_example: hanabi-learning-environment/CMakeFiles/game_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/sad/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable game_example"
	cd /sad/build/hanabi-learning-environment && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/game_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
hanabi-learning-environment/CMakeFiles/game_example.dir/build: hanabi-learning-environment/game_example

.PHONY : hanabi-learning-environment/CMakeFiles/game_example.dir/build

hanabi-learning-environment/CMakeFiles/game_example.dir/clean:
	cd /sad/build/hanabi-learning-environment && $(CMAKE_COMMAND) -P CMakeFiles/game_example.dir/cmake_clean.cmake
.PHONY : hanabi-learning-environment/CMakeFiles/game_example.dir/clean

hanabi-learning-environment/CMakeFiles/game_example.dir/depend:
	cd /sad/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /sad /sad/hanabi-learning-environment /sad/build /sad/build/hanabi-learning-environment /sad/build/hanabi-learning-environment/CMakeFiles/game_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : hanabi-learning-environment/CMakeFiles/game_example.dir/depend

