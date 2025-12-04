#!/usr/bin/env bash
set -euo pipefail

MPI_USER="mpiuser"
MPI_UID=${MPI_UID:-1000}
MPI_GID=${MPI_GID:-1000}
MPI_SHARED=${MPI_SHARED:-/opt/mpi_shared}
SSH_DIR="${MPI_SHARED}/ssh"
HOSTS_DIR="${MPI_SHARED}/hosts"
HOSTFILE="${HOSTS_DIR}/hostfile"
AUTHORIZED="${SSH_DIR}/authorized_keys"
PRIVATE="${SSH_DIR}/id_rsa"
PUBLIC="${SSH_DIR}/id_rsa.pub"
HOSTFILE_LOCK="${HOSTS_DIR}/hostfile.lock"
KEYGEN_LOCK="${SSH_DIR}/keygen.lock"

mkdir -p "${SSH_DIR}" "${HOSTS_DIR}"
chmod 2775 "${MPI_SHARED}" || true
chmod 700 "${SSH_DIR}" || true

if mkdir "${KEYGEN_LOCK}" 2>/dev/null; then
  echo "[mpi-node] Acquired key generation lock. Creating keys..."
  
  rm -f "${PRIVATE}" "${PUBLIC}"

  ssh-keygen -t rsa -b 4096 -f "${PRIVATE}" -N "" -C "mpi_shared_key" -q
  
  chmod 600 "${PRIVATE}"
  chmod 644 "${PUBLIC}"
  
  echo "[mpi-node] Keys generated successfully."
  
  rmdir "${KEYGEN_LOCK}"
else
  echo "[mpi-node] Another node is generating keys. Waiting..."
  
  while [ ! -f "${PRIVATE}" ]; do
    sleep 0.2
  done
  
  while [ ! -f "${PUBLIC}" ]; do
    sleep 0.2
  done
  
  echo "[mpi-node] Keys found. Proceeding."
fi

touch "${AUTHORIZED}"
chmod 600 "${AUTHORIZED}"
PUB_CONTENT="$(cat "${PUBLIC}")"

if ! grep -qxF "${PUB_CONTENT}" "${AUTHORIZED}" 2>/dev/null; then
  echo "${PUB_CONTENT}" >> "${AUTHORIZED}"
fi

chown -R ${MPI_UID}:${MPI_GID} "${SSH_DIR}" "${HOSTS_DIR}" || true
chmod 700 "${SSH_DIR}"
chmod 600 "${AUTHORIZED}" || true

NODE_IP="$(hostname -I | awk '{print $1}' || true)"
if [ -z "${NODE_IP}" ]; then
  NODE_IP="$(getent hosts "$(hostname)" | awk '{print $1}' || true)"
fi

if [ -n "${NODE_IP}" ]; then
  while ! mkdir "${HOSTFILE_LOCK}" 2>/dev/null; do
    sleep 0.1
  done

  touch "${HOSTFILE}"
  chown ${MPI_UID}:${MPI_GID} "${HOSTFILE}" || true

  if ! grep -qxF "${NODE_IP}" "${HOSTFILE}" 2>/dev/null; then
    echo "${NODE_IP}" >> "${HOSTFILE}"
  fi

  rmdir "${HOSTFILE_LOCK}" || true
  chown ${MPI_UID}:${MPI_GID} "${HOSTFILE}" || true
  echo "[mpi-node] Registered IP ${NODE_IP} in ${HOSTFILE}"
else
  echo "[mpi-node] WARNING: could not determine container IP"
fi

mkdir -p "/home/${MPI_USER}/.ssh"
cp -f "${AUTHORIZED}" "/home/${MPI_USER}/.ssh/authorized_keys" || true
chown -R ${MPI_UID}:${MPI_GID} "/home/${MPI_USER}/.ssh"
chmod 700 "/home/${MPI_USER}/.ssh"
chmod 600 "/home/${MPI_USER}/.ssh/authorized_keys"

ssh-keygen -A >/dev/null 2>&1 || true

echo "[mpi-node] Starting sshd..."
exec /usr/sbin/sshd -D -e