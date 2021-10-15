FROM nvidia/cudagl:11.2.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git lsb-release

RUN apt-get install libpcl-dev

WORKDIR /root/

RUN git clone https://github.com/crigroup/openrave-installation

WORKDIR /root/openrave-installation

RUN /bin/sh install-dependencies.sh \
    && /bin/sh install-osg.sh \
    && /bin/sh install-fcl.sh \
    && /bin/sh install-openrave.sh

ENV LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH"
ENV OPENTHREADS_INC_DIR="/usr/local/include"
ENV OPENTHREADS_LIB_DIR="/usr/local/lib64:/usr/local/lib"
ENV PATH="$OPENTHREADS_LIB_DIR:$PATH"

WORKDIR /root/

RUN git clone https://github.com/truncs/trajopt.git \
    && mkdir /root/trajopt/build && cd /root/trajopt/build \
    && cmake .. \
    && make -j

ENV PYTHONPATH="$PYTHONPATH:/root/trajopt/build/lib/:/root/trajopt/src/cloudproc/"
