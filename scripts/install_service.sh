#!/usr/bin/env bash
# =============================================================================
# NYXARA — install the always-on systemd service (Kali / any systemd Linux).
# -----------------------------------------------------------------------------
# Installs NYXARA (HTTP server + background mind) as a system service that:
#   * starts automatically on every boot,
#   * restarts itself within seconds if it crashes or is killed,
#   * runs 24/7 while the machine is on, before/without any login.
#
# A process cannot run while the machine is powered OFF — this delivers
# "always alive" the only way software can: auto-start on boot + auto-restart.
#
# Usage (needs root, so it can write the unit and create the service user):
#   sudo bash scripts/install_service.sh            # install + enable + start
#   sudo bash scripts/install_service.sh --uninstall
#
# Requires Python 3.11+ and that NYXARA is installed (pip install -e .) so the
# `nyxara-daemon` command exists.
# =============================================================================
set -euo pipefail

SERVICE_NAME="nyxara"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
SERVICE_USER="nyxara"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="${REPO_ROOT}/deploy/systemd/nyxara.service"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "This installer needs root (writes ${UNIT_PATH}, manages a service user)." >&2
    echo "Re-run:  sudo bash scripts/install_service.sh $*" >&2
    exit 1
  fi
}

require_systemd() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemd (systemctl) not found — this installer targets systemd Linux (Kali etc.)." >&2
    echo "On other init systems, run 'nyxara-daemon' under your own supervisor." >&2
    exit 1
  fi
}

uninstall() {
  require_root "$@"
  require_systemd
  echo "==> Disabling and removing ${SERVICE_NAME} ..."
  systemctl disable --now "${SERVICE_NAME}" 2>/dev/null || true
  rm -f "${UNIT_PATH}"
  systemctl daemon-reload
  echo "==> Removed. (The '${SERVICE_USER}' user and ~/.nyxara data were left intact.)"
  echo "    Remove the user manually if you want:  userdel ${SERVICE_USER}"
  exit 0
}

if [[ "${1:-}" == "--uninstall" || "${1:-}" == "-u" ]]; then
  uninstall "$@"
fi

require_root "$@"
require_systemd

# --- locate the daemon command -------------------------------------------------------
# Prefer an installed `nyxara-daemon`; fall back to the module form with an explicit python.
EXECSTART=""
if command -v nyxara-daemon >/dev/null 2>&1; then
  EXECSTART="$(command -v nyxara-daemon)"
else
  PYBIN="$(command -v python3 || command -v python || true)"
  if [[ -z "${PYBIN}" ]]; then
    echo "Neither 'nyxara-daemon' nor python found on PATH." >&2
    echo "Install NYXARA first:  cd ${REPO_ROOT} && python -m pip install -e ." >&2
    exit 1
  fi
  echo "==> 'nyxara-daemon' not on PATH; using: ${PYBIN} -m nyxara.daemon"
  EXECSTART="${PYBIN} -m nyxara.daemon"
fi

# --- ensure the dedicated, unprivileged service user exists --------------------------
if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
  echo "==> Creating system user '${SERVICE_USER}' ..."
  useradd --system --create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
else
  echo "==> System user '${SERVICE_USER}' already exists — reusing."
fi

# --- render the unit from the template -----------------------------------------------
if [[ ! -f "${TEMPLATE}" ]]; then
  echo "Unit template missing: ${TEMPLATE}" >&2
  exit 1
fi
echo "==> Writing ${UNIT_PATH} ..."
sed -e "s#@USER@#${SERVICE_USER}#g" \
    -e "s#@WORKDIR@#${REPO_ROOT}#g" \
    -e "s#@EXECSTART@#${EXECSTART}#g" \
    "${TEMPLATE}" > "${UNIT_PATH}"

# Let the service user read the repo (EnvironmentFile / working dir).
chmod +x "$0" 2>/dev/null || true

# --- enable + start ------------------------------------------------------------------
echo "==> Reloading systemd and enabling the service ..."
systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo
echo "Done. NYXARA is now an always-on service."
echo "  Status:    systemctl status ${SERVICE_NAME}"
echo "  Live logs: journalctl -u ${SERVICE_NAME} -f"
echo "  Restart:   systemctl restart ${SERVICE_NAME}"
echo "  Stop it:   systemctl stop ${SERVICE_NAME}      (this boot only)"
echo "  Turn off:  systemctl disable --now ${SERVICE_NAME}   (no auto-start on boot)"
echo "  Uninstall: sudo bash scripts/install_service.sh --uninstall"
echo
echo "Tip: set NYXARA_SERVER__API_TOKEN (and any LLM keys) in ${REPO_ROOT}/.env,"
echo "     then 'systemctl restart ${SERVICE_NAME}'. Keep HOST=127.0.0.1 unless you"
echo "     intend to expose the API to your network."
