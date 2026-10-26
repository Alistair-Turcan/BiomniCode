"""Verify immutable packaged source and runtime resources, optionally checking imports."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imports",action="store_true")
    args=p.parse_args()
    source=json.loads((ROOT/"provenance/source_manifest.json").read_text())
    for entry in source:
        path=ROOT/entry["path"]
        assert path.is_file() and sha256(path)==entry["sha256"],entry["path"]
    result={"protected_source_and_asset_files":len(source),"all_hashes_match":True}
    if args.imports:
        import pyscenic.aucell as ac
        import ctxcore.recovery as rec
        import importlib.metadata as metadata
        paths={"pyscenic":Path(ac.__file__).resolve(),"ctxcore":Path(rec.__file__).resolve()}
        for package,path in paths.items():
            repo="pySCENIC" if package=="pyscenic" else "ctxcore"
            assert path.is_relative_to(ROOT/repo/"src"),(package,path)
        result["local_imports"]={k:str(v.relative_to(ROOT)) for k,v in paths.items()}
        result["distribution_versions"]={k:metadata.version(k) for k in paths}
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
