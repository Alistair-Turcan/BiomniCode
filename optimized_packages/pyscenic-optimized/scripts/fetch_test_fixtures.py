"""Optionally fetch unchanged upstream test fixtures, checking exact pinned hashes.

Files are data, never executed. Consult the upstream projects/datasets for
applicable data-use terms; this script does not grant additional rights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def check(record,path):
    return path.is_file() and path.stat().st_size==record["bytes"] and digest(path)==record["sha256"]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-only",action="store_true",help="Check already downloaded fixtures; do not use the network.")
    args=p.parse_args()
    records=json.loads((ROOT/"provenance/test_fixtures.json").read_text())
    for r in records:
        dest=(ROOT/r["path"]).resolve()
        assert dest.is_relative_to(ROOT) and "/src/resources/tests/" in str(dest)
        if check(r,dest):
            print("Verified",r["path"],flush=True)
            continue
        if args.check_only:
            raise SystemExit("Missing or changed fixture: "+r["path"]+"\nRun scripts/fetch_test_fixtures.py first.")
        if dest.exists():
            raise SystemExit("Refusing to overwrite a modified fixture: "+r["path"])
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix(dest.suffix+".partial")
        request=urllib.request.Request(r["url"],headers={"User-Agent":"pyscenic-optimized-test-fixtures/1.0"})
        with urllib.request.urlopen(request,timeout=120) as response, temp.open("wb") as f:
            while block:=response.read(1<<20): f.write(block)
        if not check(r,temp):
            raise SystemExit("Fixture checksum/size mismatch; partial download retained: "+str(temp))
        temp.replace(dest)
        print("Downloaded and verified",r["path"],flush=True)
    print("All optional upstream fixtures verified.")

if __name__=="__main__": main()
