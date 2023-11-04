# start with a base dev container
FROM mcr.microsoft.com/devcontainers/base:jammy

# Configure apt and install packages in noninteractive mode
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install \

    # dev tools
    git \
    doxygen \

    # python
    python3-tk \
    python3-pip \

    # C++
    build-essential  \
    gdb  \
    cmake \

    # C++ opengl abstraction layer SFML
    libsfml-dev

# install python libraries
RUN pip install \

    # for installation
    setuptools \

    # common scientific tools
    matplotlib \
    scipy \
    numpy \
    quadprog \

    # robotics simulation
    pin \

    # robotics visualization
    meshcat

# create a symbolic linking python to python3 for conveniencce
RUN ln -s /usr/bin/python3 /usr/bin/python

# NOTE: this is how you can edit your .bashrc
#   RUN echo "some new text in your bashrc file" >> /etc/bash.bashrc
