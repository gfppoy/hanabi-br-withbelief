#!/bin/bash

if [ -d "./build" ]; then rm -Rf ./build; fi

mkdir build
cd build
cmake ..
make -j10

if [ -d "/sad_lib" ]; then rm -Rf /sad_lib; fi
mkdir /sad_lib
cp *.so /sad_lib && cp ./rela/*.so /sad_lib
#echo $PATH
#export PYTHONPATH=/sad_lib_tmp:/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
#echo $PATH
