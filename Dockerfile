FROM datajoint_local

MAINTAINER Edgar Y. Walker <edgar.walker@gmail.com>

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
    wget


# Build HDF5
RUN cd ; wget https://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.17.tar.gz \
    && tar zxf hdf5-1.8.17.tar.gz \
    && mv hdf5-1.8.17 hdf5-setup \
    &&  cd hdf5-setup \
    && ./configure --prefix=/usr/local/ \
    &&  make -j 12 && make install \
    && cd  \
    && rm -rf hdf5-setup \
    && apt-get -yq autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install OpenCV
RUN wget http://mirrors.kernel.org/ubuntu/pool/main/g/gmp/libgmp10_6.1.0+dfsg-2_amd64.deb\
	&& dpkg -i libgmp10_6.1.0+dfsg-2_amd64.deb \
	&& rm libgmp10_6.1.0+dfsg-2_amd64.deb 

RUN 	apt-get update && \
    	apt-get install -y libhogweed2 && \
	wget http://mirrors.kernel.org/ubuntu/pool/main/p/p11-kit/libp11-kit0_0.23.1-3_amd64.deb&& \
	dpkg -i libp11-kit0_0.23.1-3_amd64.deb && \
	rm libp11-kit0_0.23.1-3_amd64.deb && \
	wget http://security.ubuntu.com/ubuntu/pool/main/libt/libtasn1-6/libtasn1-6_4.5-2ubuntu0.1_amd64.deb&& \
	dpkg -i libtasn1-6_4.5-2ubuntu0.1_amd64.deb && \
	rm libtasn1-6_4.5-2ubuntu0.1_amd64.deb && \
	wget http://security.ubuntu.com/ubuntu/pool/main/g/gnutls28/libgnutls-deb0-28_3.3.8-3ubuntu3.2_amd64.deb&& \
	dpkg -i libgnutls-deb0-28_3.3.8-3ubuntu3.2_amd64.deb && \
	rm libgnutls-deb0-28_3.3.8-3ubuntu3.2_amd64.deb&& \
	wget http://security.ubuntu.com/ubuntu/pool/universe/o/openjpeg/libopenjpeg5_1.5.2-3.1_amd64.deb&& \
	dpkg -i libopenjpeg5_1.5.2-3.1_amd64.deb && \
	rm libopenjpeg5_1.5.2-3.1_amd64.deb

RUN \
  apt-get update && \
  apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && \
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


# install HDF5 reader and rabbit-mq client lib
RUN pip3 install h5py 


# Install Lucas
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


# Install pipeline
COPY . /data/pipeline
RUN \
  pip3 install -e pipeline/python/

# Get pupil tracking repo
RUN \
  git clone https://github.com/cajal/pupil-tracking.git && \
  pip3 install -e pupil-tracking/

RUN \
  pip3 install oct2py && \
  pip3 install git+https://github.com/atlab/tiffreader


ENTRYPOINT ["worker"]
  
