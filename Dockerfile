FROM ubuntu:18.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y git \
    # Pangolin dependencies
    libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols libegl1-mesa-dev libc++-dev libglew-dev libeigen3-dev cmake g++ libjpeg-dev libpng-dev libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev \
    # ORB-SLAM3 dependencies
    libboost-dev libssl-dev

RUN git clone https://gitlab.com/libeigen/eigen.git
RUN cd eigen && git checkout 3.1.0 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && make install

RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv && git checkout 4.4.0 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF && make -j$(nproc) && make install

RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
RUN cd Pangolin && git checkout v0.6 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF && make -j$(nproc) && make install

RUN git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
RUN cd ORB_SLAM3/Thirdparty/DBoW2 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
RUN cd ORB_SLAM3/Thirdparty/g2o && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
RUN cd ORB_SLAM3/Thirdparty/Sophus && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF && make -j$(nproc)
RUN cd ORB_SLAM3/Vocabulary && tar -xf ORBvoc.txt.tar.gz
RUN cd ORB_SLAM3 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make