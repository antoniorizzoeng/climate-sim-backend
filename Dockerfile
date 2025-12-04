FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates \
    openmpi-bin libopenmpi-dev \
    libyaml-cpp-dev \
    libnetcdf-dev libnetcdf-c++4-dev netcdf-bin libpnetcdf-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/antoniorizzoeng/climate-sim-mpi-cpp.git core
WORKDIR /build/core
RUN mkdir -p build
WORKDIR /build/core/build

RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
 && cmake --build . -j$(nproc) \
 && cmake --install . --prefix /install

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    openssh-server openssh-client \
    openmpi-bin libopenmpi-dev \
    libyaml-cpp-dev \
    libnetcdf-dev libnetcdf-c++4-dev netcdf-bin libpnetcdf-dev \
    libhdf5-dev \
    gosu curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=mpiuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd -g ${USER_GID} ${USERNAME} || true \
 && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME}

RUN mkdir -p /var/run/sshd \
 && mkdir -p /opt/mpi_shared/ssh /opt/mpi_shared/hosts \
 && chown -R ${USER_UID}:${USER_GID} /opt/mpi_shared

COPY --from=builder /install/bin/ /tmp/climate_bin/

RUN if [ -f /tmp/climate_bin/climate_sim ]; then \
        cp /tmp/climate_bin/climate_sim /usr/local/bin/climate_sim; \
    elif [ -f /build/core/build/climate_sim ]; then \
        cp /build/core/build/climate_sim /usr/local/bin/climate_sim; \
    else \
        echo "ERROR: climate_sim binary not found in builder output" && exit 1; \
    fi \
 && chmod 755 /usr/local/bin/climate_sim \
 && chown ${USERNAME}:${USERNAME} /usr/local/bin/climate_sim


RUN chmod 0755 /usr/local/bin/climate_sim || true \
 && chown ${USERNAME}:${USERNAME} /usr/local/bin/climate_sim || true

RUN mkdir -p /workspace && chown -R mpiuser:mpiuser /workspace
WORKDIR /workspace

COPY docker/entrypoint-api.sh /usr/local/bin/entrypoint-api.sh
COPY docker/entrypoint-worker.sh /usr/local/bin/entrypoint-worker.sh
COPY docker/entrypoint-mpi.sh /usr/local/bin/entrypoint-mpi.sh
RUN chmod +x /usr/local/bin/entrypoint-*.sh

COPY . /workspace
RUN pip3 install --no-cache-dir -r requirements.txt

USER root
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
