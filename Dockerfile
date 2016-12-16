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
    bzip2

RUN \
  apt-get update && \
  apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libglib2.0-0 && \
  apt-get install -y python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev && \
  apt-get install -y libatlas-base-dev gfortran && \
  git clone https://github.com/Itseez/opencv.git && \
  cd opencv && git checkout 3.1.0 && \
  cd .. && git clone https://github.com/Itseez/opencv_contrib.git && \
  cd opencv_contrib && git checkout 3.1.0 && \
  cd ../opencv && mkdir build && cd build && \
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
  	    -D WITH_CUDA=OFF \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/data/opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON .. && \
  make -j4 && \
  make install && \
  ldconfig && \
  rm -rf /data/opencv /data/opencv_contrib && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# --- install HDF5 reader and rabbit-mq client lib
RUN pip3 install h5py 


# --- install Spike Triggered Mixture Model for deconvolution
RUN \
  git clone https://github.com/lucastheis/cmt.git && \
  cd ./cmt/code/liblbfgs && \
  ./autogen.sh && \
  ./configure --enable-sse2 && \
  make CFLAGS="-fPIC" && \
  cd ../..  && \
  python3 setup.py build && \
  python3 setup.py install

RUN \
  pip3 install git+https://github.com/cajal/c2s.git

# --- install CaImAn

RUN git clone --recursive -b dev https://github.com/simonsfoundation/CaImAn.git && \
    pip3 install --file CaImAn/requirements_pip.txt && \
    apt-get install -y libc6-i386 libsm6 libxrender1 && \
    pip3 install pyqt=4.11.4
RUN pip3 install -e CaImAn/

# --- install tiffreader
RUN \
  pip3 install oct2py && \
  pip3 install git+https://github.com/atlab/tiffreader

# --- install pipeline
COPY . /data/pipeline
RUN \
  pip3 install -e pipeline/python/



ENTRYPOINT ["worker"]
  
