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

#   for installation
RUN pip install setuptools

#   common scientific tools
RUN pip install matplotlib
RUN pip install scipy
RUN pip install numpy

#   robotics simulation
RUN pip install pin

#   robotics visualization
RUN pip install meshcat

# create a symbolic linking python to python3 for conveniencce
RUN ln -s /usr/bin/python3 /usr/bin/python

# NOTE: this is how you can edit your .bashrc
#   RUN echo "some new text in your bashrc file" >> /etc/bash.bashrc
