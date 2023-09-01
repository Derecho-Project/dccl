#!/usr/bin/env bash
export PREFIX=$HOME/.dccl
export INSTALL_PREFIX=$PREFIX/opt

# target=Debug
target=Release

# 0 - install openmpi
wget -c https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.bz2
tar -xf openmpi-4.1.5.tar.bz2
cd openmpi-4.1.5
./configure --prefix=${INSTALL_PREFIX}
make -j `nproc`
make install
cd ..

# 1 - install derecho
sudo apt install libspdlog-dev libssl-dev libtool librdmacm-dev libibverbs-dev -y
git clone https://github.com/derecho-project/derecho.git
cd derecho
git checkout oob
scripts/prerequisites/install-mutils.sh $HOME/.dccl/opt
scripts/prerequisites/install-mutils-containers.sh $HOME/.dccl/opt
scripts/prerequisites/install-libfabric.sh $HOME/.dccl/opt
scripts/prerequisites/install-json.sh $HOME/.dccl/opt
export CMAKE_PREFIX_PATH=$HOME/.dccl/opt
export C_INCLUDE_PATH=$HOME/.dccl/opt/include/
export CPLUS_INCLUDE_PATH=$HOME/.dccl/opt/include/
export LIBRARY_PATH=$HOME/.dccl/opt/lib/:$HOME/.dccl/opt/lib64/
export LD_LIBRARY_PATH=$HOME/.dccl/opt/lib/:$HOME/.dccl/opt/lib64/
cat build.sh | sed 's/\/usr\/local/$HOME\/\.dccl\/opt/g' > my_build.sh
chmod +x my_build.sh
./my_build.sh ${target}
cd build-${target}
make install
cd ../..

# 2 - install dccl
git clone git@github.com:derecho-project/dccl.git
cd dccl
git checkout main
mkdir build-${target}
cd build-${target}
cmake -DOMPI_WINDOW=on -DCMAKE_BUILD_TYPE=${target} -DCMAKE_INSTALL_PREFIX=$HOME/.dccl/opt ..
make -j `nproc`
# make install
cd ../..
