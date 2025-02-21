#!/bin/bash

BUILD_DIR="./build"
INSTALL_PREFIX="./bin" 

mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_PREFIX"

cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -G Ninja

cmake --build "$BUILD_DIR" -j16

cmake --install "$BUILD_DIR"