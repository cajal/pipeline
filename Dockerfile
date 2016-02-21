FROM datajoint/datajoint-dev

MAINTAINER Edgar Y. Walker <edgar.walker@gmail.com>

WORKDIR /data

# install tools to compile
RUN \
  apt-get update && \
  apt-get install -y -q \
    build-essential && \
  apt-get update && \
  apt-get install -y -q \
    autoconf \
    automake \
    libtool

# Install Lucas
RUN \
  git clone https://github.com/lucastheis/cmt.git && \
  cd ./cmt/code/liblbfgs && \
  ./autogen.sh && \
  ./configure --enable-sse2 && \
  make CFLAGS="-fPIC" && \
  cd ../..  && \
  python setup.py build && \
  python setup.py install


# Install OpenCV
RUN \
  apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && \
  apt-get install -y python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev && \
  apt-get install -y libatlas-base-dev gfortran && \
  git clone https://github.com/Itseez/opencv.git && \
  cd opencv && git checkout 3.1.0 && \
  cd .. && git clone https://github.com/Itseez/opencv_contrib.git && \
  cd opencv_contrib && git checkout 3.1.0 && \
  cd ../opencv && mkdir build && cd build && \
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/data/opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON .. && \
  make -j4 && \
  make install && \
  ldconfig && \
  rm -rf /data/opencv /data/opencv_contrib

RUN \
  apt-get update && \
  apt-get install -y --fix-missing octave 


RUN \
  pip install git+https://github.com/cajal/c2s.git

COPY . /data/pipeline
RUN \
  pip install -e pipeline/python/

# Get pupil tracking repo
RUN \
  git clone https://github.com/cajal/pupil-tracking.git

RUN \
  pip install oct2py && \
  pip install git+https://github.com/atlab/tiffreader

#RUN \
#  apt-get install -y python-pip && \
#  pip2 install pandas && \
#  apt-get install -y python-scipy

ENTRYPOINT ["worker"]
  
