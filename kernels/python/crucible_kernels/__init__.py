"""C++ hot kernels for crucible.

``feitu`` decodes one feitu v3 minute dump into a dict of numpy columns holding raw vendor
values. Calls release the GIL, so a thread pool over files scales across cores.
"""

from crucible_kernels import _feitu as feitu

__all__ = ["feitu"]
