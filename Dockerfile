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

RUN \
  pip install git+https://github.com/cajal/c2s.git

COPY . /data/pipeline
RUN \
  pip install pipeline/python/
