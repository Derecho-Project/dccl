#!/usr/bin/env bash
export PREFIX=$HOME/.dccl
export INSTALL_PREFIX=$PREFIX/opt

# 1 - install open mpi
wget -c https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.bz2
tar -xf openmpi-4.1.5.tar.bz2
cd openmpi-4.1.5
./configure --prefix=${INSTALL_PREFIX}
make -j `nproc`
make install
cd ..

# 2 - build mpich
git clone https://github.com/intel/mpi-benchmarks.git
cd mpi-benchmarks
export CC=${INSTALL_PREFIX}/bin/mpicc
export CXX=${INSTALL_PREFIX}/bin/mpicxx
make clean
make all
cd ..
unset CC
unset CXX

