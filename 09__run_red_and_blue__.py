#!/usr/bin/env python3
"""
09__run_red_and_blue__.py (patched)
===================================
Driver script that runs Step 06 and Step 07 sequentially for both RED and BLUE
sets. Uses absolute paths to avoid “No such file or directory” errors,
ensuring helper scripts are found regardless of the current working directory.
"""
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable                            # current Python interpreter
ROOT = Path(__file__).resolve().parent         # folderr containing this script
SCRIPT_06 = ROOT / "06__prepare_admm_matrices__.py"
SCRIPT_07 = ROOT / "07__ADMM__solver__.py"

# Check existence of required scripts
if not SCRIPT_06.is_file():
    sys.exit(f"[FATAL] cannot find {SCRIPT_06}")
if not SCRIPT_07.is_file():
    sys.exit(f"[FATAL] cannot find {SCRIPT_07}")

for colour in ("red", "blue"):
    print("\n============================")
    print(f"[STEP] Optimising {colour.upper()} set")
    print("============================")

    # Propagate colour choice to child scripts via environment variable
    env = os.environ.copy()
    env["OPT_SET"] = colour

    # Step 06 — prepare ADMM matrices
    print("[INFO] running 06 – prepare matrices …")
    subprocess.run([PY, str(SCRIPT_06)], env=env, check=True)

    # Step 07 — run ADMM solver
    print("[INFO] running 07 – ADMM solver …")
    subprocess.run([PY, str(SCRIPT_07)], env=env, check=True)

print("\n[ALL DONE] Both RED and BLUE sets optimised ✓")
