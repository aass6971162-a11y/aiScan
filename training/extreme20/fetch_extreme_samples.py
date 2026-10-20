"""Read UVDoc's public ZIP directory and fetch selected entries with HTTP ranges."""
import argparse
import binascii
import concurrent.futures
import json
import os
from pathlib import Path
import struct
import time
import urllib.request
import zlib

URL = "https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip"
ROOT = Path(os.environ.get("SCAN_BENCH_WORK",Path(__file__).resolve().parent)) / "extreme20"

def request(start, end):
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": "DocumentScanner-Regression/1.0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                if response.status != 206:
                    raise RuntimeError(f"Range unsupported: {response.status}")
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)

def index():
    ROOT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "archive-index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=30) as response:
        size = int(response.headers["Content-Length"])
    tail = request(size - 65558, size - 1)
    pos = tail.rfind(b"PK\x05\x06")
    _, _, _, count, _, length, offset, _ = struct.unpack_from("<4s4H2IH", tail, pos)
    directory = request(offset, offset + length - 1)
    entries, pos = [], 0
    for _ in range(count):
        fields = struct.unpack_from("<4s6H3I5H2I", directory, pos)
        assert fields[0] == b"PK\x01\x02"
        name_length, extra_length, comment_length = fields[10:13]
        name = directory[pos + 46:pos + 46 + name_length].decode("utf-8")
        entries.append(dict(name=name, method=fields[4], crc=fields[7], compressed=fields[8], size=fields[9], offset=fields[16]))
        pos += 46 + name_length + extra_length + comment_length
    result = dict(source=URL, archive_bytes=size, entries=entries)
    cache.write_text(json.dumps(result, indent=2))
    return result

def fetch(entry):
    path = ROOT / entry["name"]
    if path.is_file() and path.stat().st_size == entry["size"]:
        return entry["name"]
    header = request(entry["offset"], entry["offset"] + 29)
    assert header[:4] == b"PK\x03\x04"
    name_length, extra_length = struct.unpack_from("<HH", header, 26)
    start = entry["offset"] + 30 + name_length + extra_length
    compressed = request(start, start + entry["compressed"] - 1)
    raw = zlib.decompress(compressed, -15) if entry["method"] == 8 else compressed
    assert len(raw) == entry["size"] and binascii.crc32(raw) & 0xffffffff == entry["crc"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return entry["name"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", nargs="*", default=[])
    args = parser.parse_args()
    data = index()
    if not args.fetch:
        print("\n".join(f"{x['name']} {x['size']}" for x in data["entries"][:100]))
        print("entries", len(data["entries"]))
    else:
        selected = [entry for entry in data["entries"] if any(entry["name"].startswith(prefix) for prefix in args.fetch) and not entry["name"].endswith("/")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for name in executor.map(fetch, selected):
                print(name, flush=True)
