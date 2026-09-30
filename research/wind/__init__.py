"""WindPy access for the extranet side of the PM-reproduction track.

Windows only (WindPy talks to a locally running, logged-in Wind terminal over IPC). Every
request is cached as Parquet keyed by a stable hash of the full request, and a cached request
is never sent again -- the Wind API has a monthly quota. WSL code reads the cache directly
from ``data/wind_cache`` through ``/mnt/c``.
"""

from wind.cache import CACHE_ROOT, cached_request, request_log
from wind.client import WindError, connect

__all__ = ["CACHE_ROOT", "WindError", "cached_request", "connect", "request_log"]
