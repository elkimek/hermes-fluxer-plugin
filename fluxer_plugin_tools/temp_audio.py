"""Private scratch-file helpers for the Fluxer voice scripts.

Captured voice audio is unredacted speech, so it must never be written to a
predictable, world-readable ``/tmp`` path where another local user could
pre-create a symlink or read it. These helpers keep every voice scratch file
private (``mkstemp`` opens ``0o600`` under a random name) and guarantee cleanup.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def secure_temp_audio_path(suffix: str = ".wav", *, prefix: str = "fluxer_voice_") -> Iterator[Path]:
    """Yield a private, unpredictable scratch path, then unlink it.

    ``mkstemp`` creates the file with mode ``0o600`` under a random name in the
    private temp dir, and the file is removed in a ``finally`` even when the
    caller raises — so raw speech neither lingers nor is readable by other users.
    """

    fd, raw = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    path = Path(raw)
    os.close(fd)
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
