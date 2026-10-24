"""Where an export comes from. v1: a zip file the customer uploads.

A source only fetches files into a staging folder; reading them is ``readers``' job, so a later
source (an S3 bucket, a database, a warehouse) only has to produce the same staged files.
Extraction is defensive: the archive is untrusted input.
"""

from __future__ import annotations

import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_COMPRESSED = 1 << 30  # 1 GB
MAX_UNCOMPRESSED = 8 << 30
MAX_RATIO = 200  # uncompressed / compressed, per file: beyond this it is a zip bomb
MAX_FILES = 2000
READABLE = (".csv", ".tsv", ".txt", ".xlsx")
ARCHIVES = (".zip", ".gz", ".tgz", ".tar", ".rar", ".7z", ".bz2", ".xz")


class UnsafeArchive(ValueError):
    """The archive breaks a safety rule; nothing was extracted."""


@dataclass(frozen=True)
class ZipSource:
    path: Path
    kind: str = "zip"

    def fetch(self, staging_dir: Path) -> list[Path]:
        """Extract the readable files under ``staging_dir/files``; returns them, sorted."""
        if self.path.stat().st_size > MAX_COMPRESSED:
            raise UnsafeArchive(f"the archive is larger than {MAX_COMPRESSED >> 20} MB")
        try:
            archive = zipfile.ZipFile(self.path)
        except zipfile.BadZipFile as exc:
            raise UnsafeArchive(f"not a zip file: {exc}") from exc
        with archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            check_members(members)
            out = staging_dir / "files"
            written = []
            for member in members:
                name = PurePosixPath(member.filename)
                if name.suffix.lower() not in READABLE:
                    continue
                target = out.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, open(target, "wb") as dst:
                    copied = 0
                    while chunk := src.read(1 << 20):
                        copied += len(chunk)
                        if copied > member.file_size:  # the header lied about the size
                            raise UnsafeArchive(f"{member.filename}: larger than declared")
                        dst.write(chunk)
                written.append(target)
        if not written:
            raise UnsafeArchive(f"no readable files ({', '.join(READABLE)}) in the archive")
        return sorted(written)


def check_members(members: list[zipfile.ZipInfo]) -> None:
    if len(members) > MAX_FILES:
        raise UnsafeArchive(f"{len(members)} files; at most {MAX_FILES}")
    total = 0
    for member in members:
        name = member.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
            raise UnsafeArchive(f"{member.filename}: a path outside the archive")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise UnsafeArchive(f"{member.filename}: a symbolic link")
        if path.suffix.lower() in ARCHIVES:
            raise UnsafeArchive(f"{member.filename}: an archive inside the archive")
        if member.compress_size and member.file_size / member.compress_size > MAX_RATIO:
            raise UnsafeArchive(f"{member.filename}: compression ratio too high")
        total += member.file_size
    if total > MAX_UNCOMPRESSED:
        raise UnsafeArchive(f"the archive expands beyond {MAX_UNCOMPRESSED >> 30} GB")
