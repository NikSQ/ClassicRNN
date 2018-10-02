#!/bin/bash

#SBATCH --job-name="rnn"
#SBATCH --workdir="/clusterFS/home/student/kopp13/ClassicRNN/src"
#SBATCH --output=/clusterFS/home/student/kopp13/ClassicRNN/logs/test%5a.out
#SBATCH --error=/clusterFS/home/student/kopp13/ClassicRNN/logs/test%5a.err
#SBATCH --open-mode=truncate
#SBATCH --cpus-per-task=1
#SBATCH --time=07-00
#SBATCH --mem=8G
#SBATCH --gres=gpu
#SBATCH --partition=gpu,gpu2,gpu6
#SBATCH --exclude=diannao
#SBATCH --array=0-0

#################
# configuration #
#################

# overwrite home directory for clean environment
#export HOME="/clusterFS/home/user/${USER}" # clusterFS for user
export HOME="/clusterFS/home/student/${USER}" # clusterFS for student
#export HOME="/srv/tmp/${USER}" # SSD on simulation hosts
#export HOME="/srv/${USER}" # SSD on your workstation

# miniconda base dir
_conda_base_dir="${HOME}"

# conda python environment to use
_conda_env="tensorflow"
# python version to use
_conda_python_version="3.5"
#_conda_python_version="2.7"
# python packages to install:
# conda packages
# _conda_install_packages="theano numpy pygpu matplotlib ipython jupyter jupyter_client"
_conda_install_packages="numpy pygpu matplotlib scipy mkl nose sphinx"
# pip packages
# _pip_install_packages="PySingular==0.9.1 jupyter_kernel_singular"
_pip_install_packages="parameterized"
# pip whl URL

if [ ${_conda_python_version:0:1} -eq 3 ]; then
  # python 3.5
  #_pip_install_whl="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp35-cp35m-linux_x86_64.whl" 
  #_pip_install_whl="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp35-cp35m-linux_x86_64.whl"
  _pip_install_whl="https://pypi.python.org/packages/72/e8/ff6c2b9377d52a7a6b4edaaecc4f09a43a461ff7c9091bdc30eae0836460/tensorflow_gpu-1.5.0rc1-cp35-cp35m-manylinux1_x86_64.whl"
else
  # python 2.7
  #_pip_install_whl="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp27-none-linux_x86_64.whl"
  #_pip_install_whl="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp27-none-linux_x86_64.whl"
  _pip_install_whl="https://pypi.python.org/packages/3b/10/e538bbf1a63c7aab2fa48edd2342ca14f46ac5ac686c3ac4ecc67e1d4b9a/tensorflow_gpu-1.5.0rc1-cp27-cp27mu-manylinux1_x86_64.whl"
fi

# overwrite theano flags
THEANO_FLAGS="floatX=float32,gpuarray.preallocate=0.9"

# we need clang-3.8 and THEANO_FLAGS="dnn.library_path=/usr/lib/x86_64-linux-gnu"
THEANO_FLAGS+=",dnn.library_path=/usr/lib/x86_64-linux-gnu"
# g++ binary name
# no path component bacause used as THEANO_FLAGS="cxx=${_gpp}"
_gpp="clang++-3.8"
# path to bin directory to overwrite the compiler version
# do not use a directory which is in your $PATH
# ${_gpp} will get used for the g++ symlink in the bin folder ${_bin_dir}
# this was necessary for tensorflow with cuda8 and cudnn5
#_bin_dir="bin"


# software requironments:
# apt install nvidia-smi nvidia-kernel-dkms nvidia-cuda-toolkit nvidia-driver nvidia-opencl-common
# apt install clang-3.8 libcudnn7-dev libcupti-dev


########################
# code for environment #
########################

# make shure ${HOME} exists
mkdir -p ${HOME} || exit 1

# define custom environment
# minimal $PATH for home
export PATH=${_conda_base_dir}/miniconda${_conda_python_version:0:1}/bin:${HOME}/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games


# compile version hack:
if [ -n "${_bin_dir}" ]; then
  mkdir -p "${_bin_dir}"
  _bin_dir=$(readlink -f ${_bin_dir}) # get full path
  if [ ! -s "${_bin_dir}/g++" ]; then
    ln -fs $(which ${_gpp}) ${_bin_dir}/g++
  fi
export PATH=${_bin_dir}:${PATH}
fi
THEANO_FLAGS+=",cxx=${_gpp}"

# python
# install miniconda
if [ ! -d ${_conda_base_dir}/miniconda${_conda_python_version:0:1} ]; then
  if [ ! -f Miniconda2-latest-Linux-x86_64.sh ]; then
    wget https://repo.continuum.io/miniconda/Miniconda${_conda_python_version:0:1}-latest-Linux-x86_64.sh
  fi
  chmod +x ./Miniconda${_conda_python_version:0:1}-latest-Linux-x86_64.sh
  ./Miniconda${_conda_python_version:0:1}-latest-Linux-x86_64.sh -b -f -p ${_conda_base_dir}/miniconda${_conda_python_version:0:1}
  rm ./Miniconda${_conda_python_version:0:1}-latest-Linux-x86_64.sh
  INSTALL=${INSTALL:-true}
  if [ ! -d ${_conda_base_dir}/miniconda${_conda_python_version:0:1} ]; then
    echo "ERROR: \"${_conda_base_dir}/miniconda${_conda_python_version:0:1}\" does not exist!"
    echo "       Ths means there was a problem while installing Miniconda. Maybe too little disk space?"
    exit 1
  fi
fi
# setup virtual environment
if [ ! -d "${_conda_base_dir}/miniconda${_conda_python_version:0:1}/envs/${_conda_env}" ]; then
  conda create --yes -q -n ${_conda_env} python=${_conda_python_version}
fi
# activate environment
source activate ${_conda_env}
if [ "${CONDA_DEFAULT_ENV}" != "${_conda_env}" ]; then
  echo "ERROR: unable to activate conda environment \"${_conda_env}\""
  exit 1
fi
# ensure right python version
# ${_conda_python_version} matches python version installed
# for example _conda_python_version=3.5 and python 3.5.3 installed will be ok
_python_ver_installed=$(python --version 2>&1 | awk '{print $2}')
[[ ${_python_ver_installed} =~ ^${_conda_python_version}.*$ ]] || { \
    echo "python version ${_python_ver_installed} installed but ${_conda_python_version} expected"
    echo "manual change required..."
    exit 1
}
# ensure all packages are installed
if [ -n "${INSTALL}" ]; then
  conda install --yes ${_conda_install_packages}
  for _pip_package in ${_pip_install_packages}; do
    pip install --exists-action=i ${_pip_package}
  done
  for _pip_package in ${_pip_install_whl}; do
    pip install --exists-action=i ${_pip_package}
  done
fi

# theano
# define default gpu device to use
if [ -z "${GPU}" ]; then
  THEANO_FLAGS+=",device=cuda"
else
  THEANO_FLAGS+=",device=${GPU}"
fi

# temporary compile dir
THEANO_TMPDIR=`mktemp -d`

# optimized theano flags (from Matthias Zöhrer)
THEANO_FLAGS+=",base_compiledir=${THEANO_TMPDIR}"


# print config
#echo -e "\n\nconfig:\n"
#echo "HOME=${HOME}"
#echo "PATH=${PATH}"
#echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
#echo "THEANO_FLAGS=${THEANO_FLAGS}"
#echo

####################
# START: user code #
####################

# run simple test (with theano flags)
export MKL_THREADING_LAYER=GNU
export THEANO_FLAGS="${THEANO_FLAGS}"
export PYTHONUNBUFFERED=TRUE

python3 ../exps/test.py
##################
# END: user code #
##################

# clean up
if [ -d "${THEANO_TMPDIR}" ]; then
    rm -rf "${THEANO_TMPDIR}"
fi
