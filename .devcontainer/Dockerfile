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

# Generate stubs for pinocchio, and fix the missing pinocchio_pywrap and visualize stub imports
RUN pip install pybind11-stubgen
RUN cd -P -- "$(python -c "import pinocchio; import os; print(os.path.dirname(pinocchio.__file__))")/.." \
    && pybind11-stubgen pinocchio -o . \
    && touch pinocchio/py.typed \
    && echo "from pinocchio_pywrap import *" >> pinocchio/__init__.pyi \
    && echo "from visualize import *" >> pinocchio/__init__.pyi

# NOTE: this is how you can edit your .bashrc
#   RUN echo "some new text in your bashrc file" >> /etc/bash.bashrc
