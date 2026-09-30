"""Serve DuckDB's built-in web SQL UI over the crucible catalog (read-only).

The catalog is attached read-only so the UI can never change the views or summaries.
The UI extension is fetched once from extensions.duckdb.org; its page loads assets from
ui.duckdb.org, while all queries and data stay on this machine.
Open http://localhost:4213 (port from DuckDB's ``ui_local_port`` setting).
"""

from __future__ import annotations

import time

import duckdb

CATALOG = "/work/crucible_data/catalog.duckdb"

con = duckdb.connect()
con.execute(f"ATTACH '{CATALOG}' AS crucible (READ_ONLY)")
con.execute("USE crucible")
con.execute("INSTALL ui")
con.execute("LOAD ui")
print(con.execute("CALL start_ui_server()").fetchall(), flush=True)
while True:
    time.sleep(3600)
