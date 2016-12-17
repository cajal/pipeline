FROM datajoint/datajoint:latest

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

RUN rm /usr/bin/python && ln -s /usr/bin/python2 /usr/bin/python && \
    apt-get install -y build-essential cmake pkg-config libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev

RUN git clone https://github.com/Itseez/opencv.git && \
    cd opencv && git checkout 3.1.0 && \
    cd .. && git clone https://github.com/Itseez/opencv_contrib.git && \
    cd opencv_contrib && git checkout 3.1.0 && \
    cd ../opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
  	    -D WITH_CUDA=ON \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/data/opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON .. && \
    make -j4 && \
    make install && \
    ldconfig && \
    rm -rf /data/opencv /data/opencv_contrib && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean


# --- install HDF5 reader
RUN pip3 install h5py
#
#
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
    apt-get install -y libc6-i386  libsuitesparse-dev libllvm3.8 llvm-3.8-dev && \
    export LLVM_CONFIG=/usr/lib/llvm-3.8/bin/llvm-config && \ 
    git clone --recursive https://github.com/fabiansinz/CaImAn.git && \
    pip3 install cython scikit-image ipyparallel psutil numba && \
    pip3 install -r CaImAn/requirements_pip.txt && \
    pip3 install git+https://github.com/j-friedrich/OASIS.git

RUN pip3 install -e CaImAn/

# --- install tiffreader
RUN \
  pip3 install oct2py && \
  pip3 install git+https://github.com/atlab/tiffreader

## --- install pipeline
COPY . /data/pipeline
RUN \
  pip3 install -e pipeline/python/

ENTRYPOINT ["worker"]

