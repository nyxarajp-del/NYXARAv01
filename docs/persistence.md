# Always-on NYXARA — persistent background daemon

This guide makes NYXARA run **persistently**: once installed she starts on her own,
stays alive at all times, comes back automatically after a reboot, and restarts herself
if she ever crashes or is killed — on **Kali / Linux** and on a **Windows** laptop,
system-wide (before/without any login).

## What "always alive" actually means

Software cannot run while the computer is **powered off**. So "always alive" is
delivered the only way it can be:

1. **Auto-start on every boot** — she comes up as soon as the OS does, before you log in.
2. **Auto-restart on failure** — if the process crashes or is killed, the OS restarts it
   within a few seconds.
3. **Runs 24/7 while the machine is on** — no terminal, no logged-in session required.

This is a normal, visible system service. You can inspect it, stop it, and uninstall it
at any time with the standard OS tools (shown below).

## What runs

The always-on process is `nyxara-daemon` — the HTTP/WebSocket **server** *and* the
**background mind** (`AutonomicLoop`) in one process. `nyxara-daemon` is just
`nyxara-serve` with the background mind switched on; it is exactly equivalent to:

```bash
NYXARA_SERVER__AUTONOMIC=true nyxara-serve
```

Every autonomic turn still passes the identical sovereign gates — autonomy buys horizon,
never extra power.

Relevant settings (all optional, `NYXARA_SERVER__` prefix):

| Setting | Default | Meaning |
| ------- | ------- | ------- |
| `AUTONOMIC` | `false` | start the background mind alongside the server (the daemon forces this on) |
| `AUTONOMIC_INTERVAL_S` | `30` | seconds between background-mind ticks |
| `AUTONOMIC_GROWTH_EVERY` | `0` | run a learning pass every N ticks (`0` = never) |
| `API_TOKEN` | *(unset)* | bearer credential; **set this for an exposed server** |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | bind address; use `0.0.0.0` only to expose to the network |

Set these (and any LLM keys) in the repo's `.env` on Linux, or as **Machine**
environment variables on Windows, then restart the service.

## Kali / Linux (systemd)

Prerequisite: NYXARA is installed so `nyxara-daemon` exists:

```bash
cd /path/to/NYXARAv01
python -m pip install -e .          # or ".[server]" for just the API
```

Install the service (needs root — it writes the unit and creates a dedicated,
unprivileged `nyxara` system user):

```bash
sudo bash scripts/install_service.sh
```

That renders [`deploy/systemd/nyxara.service`](../deploy/systemd/nyxara.service) into
`/etc/systemd/system/nyxara.service` with the real paths for this machine, then
`systemctl enable --now nyxara`.

Manage it:

```bash
systemctl status nyxara            # is she alive?
journalctl -u nyxara -f            # live logs (watch the loop tick)
systemctl restart nyxara           # e.g. after editing .env
systemctl stop nyxara              # stop until next boot
systemctl disable --now nyxara     # stop AND don't auto-start on boot
sudo bash scripts/install_service.sh --uninstall   # remove the unit entirely
```

**Resilience is built into the unit:** `Restart=always`, `RestartSec=3`,
`WantedBy=multi-user.target`. Kill it (`sudo systemctl kill nyxara`) and it comes right
back; reboot and it returns on its own.

> If your distro uses a non-systemd init, skip the installer and run `nyxara-daemon`
> under your own supervisor (e.g. an `/etc/init.d` script, `supervisord`, or a
> container with `--restart=always`).

## Windows laptop (SYSTEM scheduled task)

Prerequisite: Python 3.11+ with NYXARA installed (`py -m pip install -e .`).

In an **elevated** PowerShell ("Run as administrator"):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_nyxara_service_windows.ps1
```

This registers a Scheduled Task named **NYXARA** that runs at **startup** as **SYSTEM**
(so it starts before login), launches `pythonw -m nyxara.daemon` (no console window),
and is configured to **restart on failure**.

Manage it:

```powershell
Get-ScheduledTask -TaskName NYXARA | Get-ScheduledTaskInfo   # status / last run
Start-ScheduledTask -TaskName NYXARA
Stop-ScheduledTask  -TaskName NYXARA
powershell -ExecutionPolicy Bypass -File scripts\install_nyxara_service_windows.ps1 -Uninstall
```

> **Alternative — a true Windows service:** install [NSSM](https://nssm.cc/) and run
> `nssm install NYXARA "<path>\pythonw.exe" "-m nyxara.daemon"`. NSSM gives you native
> service semantics (auto-restart, recovery actions) if you prefer that over a task.

## Verifying it survives everything

- **Crash/kill:** kill the process — it should be back within seconds
  (Linux: `sudo systemctl kill nyxara`; Windows: end the `pythonw` task in Task Manager).
- **Reboot:** restart the machine and, without logging in, confirm she is listening:
  `curl -s http://127.0.0.1:8000/health` → `200`.
- **Logs:** Linux `journalctl -u nyxara -f`; Windows check Task Scheduler → NYXARA →
  History, or point the server logs at a file via your env config.

## Security note

An always-on server is reachable for as long as the machine is on. **Always set
`NYXARA_SERVER__API_TOKEN`** to a strong secret, and keep the bind host at `127.0.0.1`
unless you deliberately want the API reachable from other machines (in which case set
`HOST=0.0.0.0` *and* a token, and firewall the port). This service is intended to run
NYXARA on **your own** machines — it is a visible, standard service, not a hidden one.
