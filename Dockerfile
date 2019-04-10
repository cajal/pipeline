#FROM datajoint/datajoint:latest
FROM ninai/pipeline:latest
LABEL maintainer="Edgar Y. Walker, Fabian Sinz, Erick Cobos, Donnie Kim"

WORKDIR /data

# Install commons
RUN git clone https://github.com/atlab/commons.git && \
    pip3 install commons/python && \
    rm -r commons

# Install pipeline
COPY . /data/pipeline
RUN pip3 install -e pipeline/python/

####### Install Deeplabcut and its dependencities #########

# Install DeepLabCut
RUN pip3 install deeplabcut==2.0.5
RUN pip3 install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04/wxPython-4.0.3-cp36-cp36m-linux_x86_64.whl

# Install dependencies of wxPython
#RUN sudo apt-get install -y libgtk2.0-dev libgtk-3-dev \
#    libjpeg-dev libtiff-dev \
#    libsdl1.2-dev libgstreamer-plugins-base1.0-dev \
#    libnotify-dev freeglut3 freeglut3-dev libsm-dev \
#    libwebkitgtk-dev libwebkitgtk-3.0-dev

# Uninstall tensorflow and install tensorflow-gpu
RUN pip3 uninstall -y tensorflow
RUN pip3 install tensorflow-gpu==1.11.0

ENTRYPOINT ["/bin/bash"]
