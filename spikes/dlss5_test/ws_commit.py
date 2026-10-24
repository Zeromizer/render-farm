"""Commit the modified assets inside a TEST workspace copy (never the live one) so the farm's git clone sees them."""
import os, subprocess, sys
ws, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
assert os.path.normcase(ws).startswith(os.path.normcase(r"C:\rp-workspace-tests")), "test workspaces only"
g = lambda *a: subprocess.run(["git", "-C", ws, *a], capture_output=True, text=True)
res = [g("status", "--short").stdout[-3000:], g("rev-parse", "--abbrev-ref", "HEAD").stdout]
g("add", "-A", "assets")
c = g("-c", "user.name=dlss5-spike", "-c", "user.email=spike@local", "commit", "-m", "dlss5 v18 clean-clip test")
res += [c.stdout[-2000:], c.stderr[-2000:], g("rev-parse", "HEAD").stdout, g("log", "--oneline", "-3").stdout]
open(os.path.join(out, "commit.txt"), "w").write("\n---\n".join(res)); print("\n---\n".join(res))
