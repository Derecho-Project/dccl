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

# 3 - install derecho
git clone https://github.com/derecho-project/derecho.git
cd derecho
scripts/prerequisites/install-mutils.sh $HOME/.dccl/opt
scripts/prerequisites/install-mutils-containers.sh $HOME/.dccl/opt
scripts/prerequisites/install-mutils-tasks.sh $HOME/.dccl/opt
scripts/prerequisites/install-libfabric.sh $HOME/.dccl/opt
scripts/prerequisites/install-json.sh $HOME/.dccl/opt
export CMAKE_PREFIX_PATH=$HOME/.dccl/opt
export C_INCLUDE_PATH=$HOME/.dccl/opt/include/
export CPLUS_INCLUDE_PATH=$HOME/.dccl/opt/include/
export LIBRARY_PATH=$HOME/.dccl/opt/lib/:$HOME/.dccl/opt/lib64/
export LD_LIBRARY_PATH=$HOME/.dccl/opt/lib/:$HOME/.dccl/opt/lib64/
cat build.sh | sed 's/\/usr\/local/$HOME\/\.dccl\/opt/g' > my_build.sh
chmod +x my_build.sh
./my_build.sh Release
cd build-Release
make install
cd ../..

# 4 - install dccl
git clone git@github.com:derecho-project/dccl.git
cd dccl
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.dccl/opt ..
make -j `nproc`
make install
cd ../..

# 5 - build Peter's mpich version
git clone https://github.com/ptwu/mpi-benchmarks.git dccl-mpi-benchmarks
cd dccl-mpi-benchmarks 
export CC=${INSTALL_PREFIX}/bin/mpicc
export CXX=${INSTALL_PREFIX}/bin/mpicxx
make clean
make all
cd ..
unset CC
unset CXX
