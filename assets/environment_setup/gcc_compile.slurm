#!/bin/bash
#SBATCH --job-name=compile_gcc
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=gcc_compile_%j.log
#SBATCH --error=gcc_compile_%j.err

# Exit immediately if any step fails
set -e

echo "========================================="
echo "Starting GCC compilation at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Allocated cores: $SLURM_CPUS_PER_TASK"
echo "========================================="

# 1. Setup build zone
BUILD_ZONE=$HOME/gcc-build-zone
mkdir -p "$BUILD_ZONE"
cd "$BUILD_ZONE"

# 2. Download and extract source
echo "Downloading GCC 11.3.0..."
wget -q https://ftp.gnu.org/gnu/gcc/gcc-11.3.0/gcc-11.3.0.tar.gz
tar -xf gcc-11.3.0.tar.gz

# 3. Download math prerequisites
cd gcc-11.3.0
echo "Fetching prerequisites (gmp, mpfr, mpc)..."
./contrib/download_prerequisites
cd ..

# 4. Configure out-of-source build
mkdir -p build-directory
cd build-directory

echo "Configuring build parameters..."
../gcc-11.3.0/configure \
    --prefix=$HOME/local/gcc11 \
    --enable-languages=c,c++ \
    --disable-multilib \
    --enable-checking=release

# 5. Compile and Install using Slurm allocated threads
echo "Compiling GCC (this will take some time)..."
make -j $SLURM_CPUS_PER_TASK

echo "Installing binaries to $HOME/local/gcc11..."
make install

# 6. Cleanup build files to save quota space
echo "Cleaning up heavy build directories..."
rm -rf "$BUILD_ZONE"

echo "========================================="
echo "GCC compilation completed successfully at $(date)"
echo "========================================="
