#!/usr/bin/env python3
"""Download the official ChEMBL PostgreSQL release with resumable verified ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import threading
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/raw/release"
NAME = "chembl_37_postgresql.tar.gz"
URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/" + NAME
SIZE = 2088889426
SHA256 = "d5fb08a6c9bf6cf7319dbdf74bdb5a7d4ce767aace670250ba1435cab4d58c58"
CHUNK = 8 * 1024 * 1024
LOCAL = threading.local()


def part(start):
    end = min(start + CHUNK, SIZE) - 1
    path = DEST / "parts" / f"{start:012d}"
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(10):
        try:
            with LOCAL.session.get(URL, headers={"Range": f"bytes={start}-{end}"},
                                   stream=True, timeout=(10, 40)) as r:
                r.raise_for_status()
                if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                    raise ValueError("Server did not honor the requested byte range")
                tmp = Path(str(path) + ".tmp")
                with open(tmp, "wb") as f:
                    for block in r.iter_content(1024 * 1024):
                        f.write(block)
                if tmp.stat().st_size != end - start + 1:
                    raise ValueError("Truncated range")
                tmp.replace(path)
                return path
        except (requests.RequestException, ValueError) as e:
            print(f"range {start}: retry {attempt + 1}: {type(e).__name__}", flush=True)
            if attempt == 9:
                raise
            time.sleep(min(2 ** attempt, 15))


def main():
    (DEST / "parts").mkdir(parents=True, exist_ok=True)
    archive = DEST / NAME
    if archive.exists():
        with open(archive, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == SHA256:
                print("Release archive already verified", flush=True)
                return
        raise RuntimeError("Existing archive checksum mismatch")
    starts = list(range(0, SIZE, CHUNK))
    t = time.monotonic()
    with ThreadPoolExecutor(12) as pool:
        for n, future in enumerate(as_completed([pool.submit(part, s) for s in starts]), 1):
            future.result()
            if n % 10 == 0 or n == len(starts):
                print(f"Downloaded {n}/{len(starts)} ranges; elapsed {time.monotonic() - t:.0f}s", flush=True)
    tmp = Path(str(archive) + ".tmp")
    digest = hashlib.sha256()
    with open(tmp, "wb") as out:
        for start in starts:
            with open(DEST / "parts" / f"{start:012d}", "rb") as f:
                for b in iter(lambda: f.read(1024 * 1024), b""):
                    out.write(b)
                    digest.update(b)
    if digest.hexdigest() != SHA256:
        raise RuntimeError("Official release SHA-256 mismatch")
    tmp.replace(archive)
    shutil.rmtree(DEST / "parts")
    print(f"Verified official release: {archive}", flush=True)


if __name__ == "__main__":
    main()
