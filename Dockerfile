FROM datajoint/datajoint-dev

MAINTAINER Fabian Sinz <sinz@bcm.edu>

WORKDIR /data

RUN \
  apt-get update && \
  apt-get install -y -q \
    build-essential && \
  apt-get update && \
  apt-get install -y -q \
    autoconf \
    automake \
    libtool

RUN \
  git clone https://github.com/lucastheis/cmt.git && \
  cd ./cmt/code/liblbfgs && \
  ./autogen.sh && \
  ./configure --enable-sse2 && \
  make CFLAGS="-fPIC" && \
  cd ./cmt && \
  python setup.py build && \
  python setup.py install

RUN \
  pip install git+https://github.com/cajal/c2s.git

RUN \
  git clone https://github.com/cajal/pipeline && \
  pip install pipeline/python/
