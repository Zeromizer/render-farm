"""Copy named files out of test workspaces on the PC into the job output (read-only on the sources)."""
import os, shutil, sys
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for i, src in enumerate(sys.argv[2:]):
    assert os.path.normcase(src).startswith((os.path.normcase(r"C:\rp-workspace-tests"), os.path.normcase(r"C:\rp-smooth"))), src
    shutil.copy2(src, os.path.join(out, f"{i}-" + os.path.basename(src)))
    print("PROGRESS", 10 + 80 * (i + 1) // len(sys.argv[2:]), src, os.path.getsize(src), flush=True)
