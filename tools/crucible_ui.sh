#!/usr/bin/env bash
# Serve the crucible overview notebook (marimo) and the DuckDB SQL UI from WSL as systemd
# user services, so they survive the shell that started them and restart on failure.
# Open them in any Windows browser -- WSL runs in mirrored networking, so localhost works.
#
#   tools/crucible_ui.sh install   # write the unit files, enable linger, start both (once)
#   tools/crucible_ui.sh start|stop|restart|status|logs
#   tools/crucible_ui.sh refresh   # stop, rebuild catalog.duckdb (research/catalog.py), start
#
#   overview : http://localhost:2718     DuckDB SQL UI : http://localhost:4213
set -uo pipefail
REPO=/mnt/c/Users/zzz/Desktop/Repos/crucible
PY=/work/crucible/venv/bin/python
UNITS=(crucible-overview crucible-duckdb-ui)
UDIR="$HOME/.config/systemd/user"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

write_units() {
  mkdir -p "$UDIR"
  cat > "$UDIR/crucible-overview.service" <<EOF
[Unit]
Description=crucible overview notebook (marimo) on :2718

[Service]
WorkingDirectory=$REPO
Environment=PYTHONPATH=$REPO/src
ExecStart=$PY -m marimo run research/notebooks/overview.py --host 127.0.0.1 --port 2718 --headless
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  cat > "$UDIR/crucible-duckdb-ui.service" <<EOF
[Unit]
Description=DuckDB SQL UI over the crucible catalog on :4213

[Service]
WorkingDirectory=$REPO
ExecStart=$PY $REPO/tools/duckdb_ui.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
}

case "${1:-status}" in
  install)
    write_units
    # keep user services running without an open session (otherwise logind kills them)
    loginctl show-user "$USER" 2>/dev/null | grep -q '^Linger=yes' || sudo loginctl enable-linger "$USER"
    systemctl --user daemon-reload
    systemctl --user enable --now "${UNITS[@]}"
    systemctl --user --no-pager status "${UNITS[@]}" | grep -E "●|Active:"
    ;;
  start|stop|restart) systemctl --user "$1" "${UNITS[@]}" ;;
  refresh)
    # the services hold catalog.duckdb open read-only; DuckDB needs them gone to write it
    systemctl --user stop "${UNITS[@]}"
    (cd "$REPO" && PYTHONPATH="$REPO/src" "$PY" research/catalog.py) || echo "catalog build FAILED"
    systemctl --user start "${UNITS[@]}"
    ;;
  status)
    systemctl --user --no-pager status "${UNITS[@]}" | grep -E "●|Active:"
    ss -ltn | grep -E ":(2718|4213) " || echo "(no listener on 2718/4213)"
    ;;
  logs) journalctl --user -n 40 --no-pager -u crucible-overview -u crucible-duckdb-ui ;;
  *) echo "usage: $0 install|start|stop|restart|refresh|status|logs"; exit 2 ;;
esac
