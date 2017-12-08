FROM ninai/pipeline:latest

MAINTAINER Fabian Sinz <sinz@bcm.edu>


ADD requirements.txt /requirements.txt

RUN pip install -r /requirements.txt \
    && rm /requirements.txt

ADD jupyter/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

#VOLUME /notebooks

WORKDIR /notebooks

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["jupyter notebook"]
