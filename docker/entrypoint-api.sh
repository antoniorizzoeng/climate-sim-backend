#!/usr/bin/env bash
set -euo pipefail

MPI_USER="mpiuser"
MPI_SHARED=${MPI_SHARED:-/opt/mpi_shared}
SSH_SRC="${MPI_SHARED}/ssh/id_rsa"

if [ -f "${SSH_SRC}" ]; then
  mkdir -p /home/${MPI_USER}/.ssh
  cp -f "${SSH_SRC}" /home/${MPI_USER}/.ssh/id_rsa
  chown ${MPI_USER}:${MPI_USER} /home/${MPI_USER}/.ssh /home/${MPI_USER}/.ssh/id_rsa || true
  chmod 700 /home/${MPI_USER}/.ssh || true
  chmod 600 /home/${MPI_USER}/.ssh/id_rsa || true
  echo "[api] SSH key copied to /home/${MPI_USER}/.ssh/id_rsa"
fi

/usr/sbin/sshd || true

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
