#!/usr/bin/env bash
source common_env.sh

export PREFIX=$HOME/${BENCHMARK_WORKSPACE}
export INSTALL_PREFIX=$PREFIX/opt

# 0 - install basic things
sudo apt update
sudo apt install libtool-bin libspdlog-dev libssl-dev -y

# 1 - install open mpi
wget -c https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.bz2
tar -xf openmpi-4.1.5.tar.bz2
cd openmpi-4.1.5
./configure --prefix=${INSTALL_PREFIX}
make -j `nproc`
make install
cd ..

# 2 - build mpich benchmark
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
git checkout oob
scripts/prerequisites/install-mutils.sh ${PREFIX}/opt
scripts/prerequisites/install-mutils-containers.sh ${PREFIX}/opt
scripts/prerequisites/install-libfabric.sh ${PREFIX}/opt
scripts/prerequisites/install-json.sh ${PREFIX}/opt
export CMAKE_PREFIX_PATH=${PREFIX}/opt
export C_INCLUDE_PATH=${PREFIX}/opt/include/
export CPLUS_INCLUDE_PATH=${PREFIX}/opt/include/
export LIBRARY_PATH=${PREFIX}/opt/lib/:${PREFIX}/opt/lib64/
export LD_LIBRARY_PATH=${PREFIX}/opt/lib/:${PREFIX}/opt/lib64/
echo "#!/usr/bin/env bash" >> my_build.sh
echo "source ../common_env.sh" >> my_build.sh
cat build.sh | sed 's/\/usr\/local/$HOME\/${BENCHMARK_WORKSPACE}\/opt/g' >> my_build.sh
chmod +x my_build.sh
./my_build.sh Release
cd build-Release
make install
cd ../..

# 4 - install dccl
ssh-keyscan github.com >> ~/.ssh/known_hosts
git clone git@github.com:derecho-project/dccl.git
cd dccl
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX}/opt ..
make -j `nproc`
make install
cd ../..

# 5 - install peer's host
for h in `cat myhostfile`
do
    ssh-keyscan $h >> ~/.ssh/known_hosts
done
