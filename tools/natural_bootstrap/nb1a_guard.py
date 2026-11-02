"""Natural Bootstrap-1a: an allowlist file-open guard (truth firewall) for the three separated paths.

Contract: docs/natural-bootstrap/nb1a-range-connectivity-contract.md, section 3.  While a guard is
active, a process-wide audit hook sees every ``open`` and directory listing.  It records each one, and
raises ``PermissionError`` for anything outside the allowlist:
- reads: the declared data files, plus code under the interpreter / repository roots;
- writes: under the declared output directories.
Audit hooks cannot be removed, so the hook is installed once and consults the active guard.  It does not
import any controller module.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

CODE_SUFFIXES = (".py", ".pyc", ".so", ".pth", ".typed")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _code_roots() -> list[str]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.path.realpath(sys.executable)),
             str(Path(__file__).resolve().parents[1])}
    for p in sys.path:
        if p and os.path.isdir(p):
            roots.add(os.path.realpath(p))
    return sorted(os.path.realpath(r) for r in roots if r)


class OpenGuard:
    _installed = False
    _active: "OpenGuard | None" = None

    def __init__(self, name: str, allow_read, allow_write_dirs) -> None:
        self.name = name
        self.allow_read = {os.path.realpath(os.fspath(p)) for p in allow_read}
        self.allow_write_dirs = [os.path.realpath(os.fspath(d)) for d in allow_write_dirs]
        self.code_roots = _code_roots()
        self.events: list[dict] = []
        self.violations: list[dict] = []

    def __enter__(self) -> "OpenGuard":
        if not OpenGuard._installed:
            sys.addaudithook(OpenGuard._hook)
            OpenGuard._installed = True
        if OpenGuard._active is not None:
            raise RuntimeError("a guard is already active")
        OpenGuard._active = self
        return self

    def __exit__(self, *exc) -> None:
        OpenGuard._active = None

    def mark(self, label: str) -> None:
        """Record an ordered event (e.g. 'freeze_verified') between file opens."""
        self.events.append({"event": "mark", "label": label})

    def _under(self, path: str, roots) -> bool:
        return any(path == r or path.startswith(r + os.sep) for r in roots)

    def classify(self, event: str, path: str, mode, flags) -> tuple[str, bool]:
        if event in ("os.listdir", "os.scandir"):
            return "listdir", self._under(path, self.code_roots) or self._under(path, self.allow_write_dirs)
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool((flags or 0) & _WRITE_FLAGS)
        if write:
            return "write", self._under(path, self.allow_write_dirs)
        if path in self.allow_read:
            return "data-read", True
        if path.endswith(CODE_SUFFIXES) and self._under(path, self.code_roots):
            return "code-read", True
        if self._under(path, self.allow_write_dirs):   # re-reading this path's own outputs
            return "own-output-read", True
        return "data-read", False

    @staticmethod
    def _hook(event: str, args: tuple) -> None:
        g = OpenGuard._active
        if g is None or event not in ("open", "os.listdir", "os.scandir"):
            return
        path = args[0] if args else None
        if path is None or isinstance(path, int):
            return
        full = os.path.realpath(os.path.abspath(os.fsdecode(path)))
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        kind, ok = g.classify(event, full, mode, flags)
        rec = {"event": event, "path": full, "kind": kind, "allowed": ok}
        g.events.append(rec)
        if not ok:
            g.violations.append(rec)
            raise PermissionError(f"NB1a guard [{g.name}]: forbidden {kind}: {full}")

    def record(self) -> dict:
        return {"guard": self.name, "allow_read": sorted(self.allow_read), "allow_write_dirs": self.allow_write_dirs,
                "code_roots": self.code_roots, "events": self.events, "violations": self.violations,
                "data_reads": sorted({e["path"] for e in self.events if e.get("kind") == "data-read"}),
                "writes": sorted({e["path"] for e in self.events if e.get("kind") == "write"})}
