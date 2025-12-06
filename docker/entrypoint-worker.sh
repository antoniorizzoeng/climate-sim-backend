#!/usr/bin/env bash
set -euo pipefail

MPI_USER="mpiuser"
MPI_SHARED=${MPI_SHARED:-/opt/mpi_shared}
SSH_SRC="${MPI_SHARED}/ssh/id_rsa"

mkdir -p "${MPI_SHARED}/ssh"

if [ -f "${SSH_SRC}" ]; then
  mkdir -p /home/${MPI_USER}/.ssh
  cp -f "${SSH_SRC}" /home/${MPI_USER}/.ssh/id_rsa
  chown ${MPI_USER}:${MPI_USER} /home/${MPI_USER}/.ssh /home/${MPI_USER}/.ssh/id_rsa || true
  chmod 700 /home/${MPI_USER}/.ssh || true
  chmod 600 /home/${MPI_USER}/.ssh/id_rsa || true
  echo "[worker] SSH key copied to /home/${MPI_USER}/.ssh/id_rsa"
fi

/usr/sbin/sshd || true

if command -v gosu >/dev/null 2>&1; then
  echo "[worker] running celery as ${MPI_USER} via gosu"
  exec gosu ${MPI_USER} /usr/local/bin/celery -A app.tasks worker --loglevel=info
else
  echo "[worker] gosu not found, falling back to su -c"
  exec su -s /bin/bash - ${MPI_USER} -c "/usr/local/bin/celery -A app.tasks worker --loglevel=info"
fi
