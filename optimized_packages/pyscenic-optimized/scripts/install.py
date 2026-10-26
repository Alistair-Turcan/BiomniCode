"""Install the two local, validated packages without altering protected source."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env",type=Path,default=ROOT/".venv")
    args=p.parse_args()
    uv=shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required. Install it using the official Astral uv instructions, then rerun.")
    env=args.env.resolve()
    py=env/"bin/python"
    if not env.exists():
        subprocess.run([uv,"venv","--python","3.11",str(env)],check=True)
    if not py.is_file():
        raise SystemExit("Expected a POSIX Python virtual environment: "+str(env))
    version=subprocess.check_output([str(py),"-c","import sys; print('.'.join(map(str,sys.version_info[:2])))"],text=True).strip()
    if version!="3.11":
        raise SystemExit("This reproducible installation requires Python 3.11, not "+version)
    subprocess.run([uv,"pip","install","--python",str(py),"-r",str(ROOT/"requirements.lock.txt")],check=True)
    versions={}
    for repo,package in [("ctxcore","ctxcore"),("pySCENIC","pyscenic")]:
        f=ROOT/repo/"src"/package/"_version.py"
        if f.exists(): versions[f]=f.read_bytes()
    build_env={**os.environ,"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CTXCORE":"0.2.1.dev23+gf76e0cf48.d20261014"}
    try:
        # Versioneer otherwise mistakes the monorepo name for a release version.
        # Use its supported generated metadata format only while building, then
        # restore the original module bytes even if the build fails.
        metadata={"version":"0+unknown","full-revisionid":"06bafba412792f6efa5a552a23bb221cc3bdea1b",
                  "dirty":True,"error":None,"date":None}
        generated="import json\n\nversion_json = '''\n"+json.dumps(metadata)+"\n'''  # END VERSION_JSON\n\ndef get_versions():\n    return json.loads(version_json)\n"
        (ROOT/"pySCENIC/src/pyscenic/_version.py").write_text(generated)
        subprocess.run([uv,"pip","install","--python",str(py),"--no-deps","--no-build-isolation",
                        "-e",str(ROOT/"ctxcore"),"-e",str(ROOT/"pySCENIC")],env=build_env,check=True)
    finally:
        for f,data in versions.items(): f.write_bytes(data)
    subprocess.run([str(py),str(ROOT/"scripts/verify_source.py"),"--imports"],check=True)
    print("\nInstalled both local optimized packages. Activate with:\n  source "+str(env/"bin/activate"))

if __name__=="__main__": main()
