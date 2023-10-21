FROM mcr.microsoft.com/devcontainers/base:jammy

# Configure apt and install packages in noninteractive mode
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install \

    # tools
    git \

    # python
    python3-tk \
    python3-pip \

    # C++
    build-essential  \
    gdb  \
    cmake

# install python libraries
RUN pip install setuptools
RUN pip install matplotlib
RUN pip install scipy
RUN pip install numpy

# create a symbolic linking python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# this is how you can edit your .bashrc
# RUN echo "some new text in your bashrc file" >> /etc/bash.bashrc
