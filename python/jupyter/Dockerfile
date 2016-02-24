#FROM at-docker:5000/pipeline:latest
FROM pipeline

MAINTAINER Fabian Sinz <sinz@bcm.edu>


# Build HDF5
RUN cd ; wget https://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.16.tar.gz \
    && tar zxf hdf5-1.8.16.tar.gz \
    && mv hdf5-1.8.16 hdf5-setup \
    &&  cd hdf5-setup \
    && ./configure --prefix=/usr/local/ \
    &&  make -j 12 && make install \
    && cd  \
    && rm -rf hdf5-setup \
    && apt-get -yq autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD requirements.txt /requirements.txt


RUN pip install -r /requirements.txt \
    && rm /requirements.txt

ADD notebooks/ /notebooks/

ADD jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

VOLUME /notebooks

WORKDIR /notebooks


ENTRYPOINT ["/bin/sh", "-c"]
CMD ["jupyter notebook"]
