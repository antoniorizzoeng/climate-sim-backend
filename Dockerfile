FROM python:3.12-slim-bookworm AS engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake curl g++ make libopenmpi-dev libpnetcdf-dev libnetcdf-dev libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*
ARG ENGINE_REF=main
WORKDIR /src
RUN curl -fsSL "https://codeload.github.com/antoniorizzoeng/climate-sim-mpi-cpp/tar.gz/${ENGINE_REF}" \
    | tar xz --strip-components=1 \
    && cmake -S . -B build -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target climate_sim --parallel 2

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg openmpi-bin libpnetcdf0d libyaml-cpp0.7 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home api && install -d -o api -g api /data
COPY --from=engine /src/build/src/climate_sim /usr/local/bin/climate_sim
COPY --from=engine /src/build/climate-sim-mpi-cppConfigVersion.cmake /usr/local/lib/cmake/climate-sim-mpi-cpp/
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api ./api
COPY app ./app
ENV DATA_DIR=/data ENGINE_BIN=/usr/local/bin/climate_sim MPIRUN_ARGS=--oversubscribe OMPI_MCA_plm=isolated
USER api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
