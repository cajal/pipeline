#FROM datajoint/datajoint:latest
FROM eywalker/tensorflow-jupyter:v1.0.1-cuda8.0-cudnn5
MAINTAINER Edgar Y. Walker, Fabian Sinz

WORKDIR /data

# install tools to compile
RUN \
  apt-get update && \
  apt-get install -y -q \
    build-essential && \
  apt-get update && \
  apt-get install  --fix-missing -y -q \
    autoconf \
    automake \
    libtool \
    octave \
    wget \
    bzip2 \
    git

RUN apt-get install -y build-essential cmake pkg-config libjpeg8-dev libtiff5-dev libjasper-dev \
    libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev ffmpeg locate libhdf5-dev && updatedb



# because opencv 3.1.0 is currently not cuda8.0 compatible, we use an
# alternative versiion 
#RUN git clone https://github.com/Itseez/opencv.git && \
#    cd opencv && git checkout 3.1.0 && \
RUN git clone https://github.com/daveselinger/opencv && \
    cd opencv && git checkout 3.1.0-with-cuda8 && \ 
    cd .. && git clone https://github.com/Itseez/opencv_contrib.git && \
    cd opencv_contrib && git checkout 3.1.0 && \
    cd ../opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE  \
    	  -D WITH_CUDA=ON         \
	  -D CMAKE_INSTALL_PREFIX=/usr/local \
	  -D OPENCV_EXTRA_MODULES_PATH=/data/opencv_contrib/modules  \
	  -D CUDA_CUDA_LIBRARY=`locate libcuda.so`  \
	  -D BUILD_EXAMPLES=ON ..  && \
    make -j12 && \
    make install && \
    ldconfig && \
    rm -rf /data/opencv /data/opencv_contrib && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean


# install datajoint
RUN pip3 install git+https://github.com/datajoint/datajoint-python.git


# --- install HDF5 reader and nose
RUN pip3 install h5py nose

## --- install Spike Triggered Mixture Model for deconvolution
RUN git clone https://github.com/lucastheis/cmt.git && \
  cd ./cmt/code/liblbfgs && \
  ./autogen.sh && \
  ./configure --enable-sse2 && \
  make CFLAGS="-fPIC" && \
  cd ../..  && \
  python3 setup.py build && \
  python3 setup.py install

RUN pip3 install git+https://github.com/cajal/c2s.git


# --- install CaImAn
RUN apt-get update -y -q && \
    apt-get install -y software-properties-common && \
    wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-3.9 main" && \
    apt-get update -y -q && \
    apt-get install -y clang-3.9 lldb-3.9 && \
    apt-get install -y libc6-i386  libsuitesparse-dev && \
    export LLVM_CONFIG=/usr/lib/llvm-3.9/bin/llvm-config && \ 
    git clone --recursive https://github.com/ecobost/CaImAn.git && \
    pip3 install cython scikit-image ipyparallel psutil numba && \
    pip3 install -r CaImAn/requirements_pip.txt && \
    pip3 install git+https://github.com/j-friedrich/OASIS.git && \
    pip3 install future cvxpy

RUN grep -vwE "install_requires=" CaImAn/setup.py > tmp && mv tmp CaImAn/setup.py &&\
    pip3 install -e CaImAn/

# --- install scanreader
RUN \
  git clone https://github.com/atlab/scanreader.git && \
  pip3 install -e scanreader

## --- install pipeline
COPY . /data/pipeline
RUN \
  pip3 install -e pipeline/python/

RUN git clone https://github.com/atlab/commons.git && \
    pip3 install -e commons/python && \
    pip3 install scikit-learn --upgrade && \
    pip3 install imreg_dft

RUN pip3 install slacker

ENTRYPOINT ["worker"]
