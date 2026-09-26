"""Build the matte venv and the TensorRT engines, so no user job pays for either.

  python worker/matte/warmup_trt.py [model ...] [--force]

TensorRT is opt-in (MATTE_TRT=1 in the worker .env, see matte.py 4.); this
script builds regardless. Run with the worker's own python (.venv) while the
farm is idle, and BEFORE the worker restarts onto a requirements.txt that
re-keys the matte venv. Default models are the two `subject` maps to
(birefnet-portrait, birefnet-general-lite); any other model runs capped CUDA
until it is warmed up here.

Each engine build takes minutes and most of the GPU. matte.py only offers
TensorRT to a job when <cache>/trt/<model>/ready.json matches the current
model file, onnxruntime and TensorRT versions (matte.trt_fingerprint), so a
stale or missing engine means capped CUDA, never a build inside a job.

The engines are built through matte.make_session(), the same function a job
calls: the cache key onnxruntime derives from the session options only matches
if the options are identical.
"""
import glob
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS = ["birefnet-portrait", "birefnet-general-lite"]


def outer(models, force):
    sys.path.insert(0, os.path.dirname(HERE))  # worker/, for venvs + proc
    import subprocess

    from venvs import venv_python

    run_kw = {"on_line": print, "cancel_check": lambda: False, "timeout_seconds": 3600}
    py = venv_python(os.path.join(HERE, "requirements.txt"), print, run_kw)
    print(f"matte venv: {os.path.basename(os.path.dirname(os.path.dirname(py)))} ({py})", flush=True)
    cmd = [py, "-u", os.path.abspath(__file__), "--inside", *models] + (["--force"] if force else [])
    return subprocess.run(cmd, cwd=HERE).returncode


def inside(models, force):
    sys.path.insert(0, HERE)
    os.environ["MATTE_TRT"] = "1"  # building engines is the point, whatever the worker default
    import matte
    from PIL import Image
    from rembg import remove

    matte.require_cuda()
    if matte._trt_libs_error:
        print(f"TensorRT unavailable: {matte._trt_libs_error}", flush=True)
        return 1
    failed = 0
    for model in models:
        d = os.path.join(matte.TRT_ROOT, model)
        ok, why = matte.trt_ready(model)
        if ok and not force:
            print(f"{model}: engine ready, skipping ({d})", flush=True)
            continue
        print(f"{model}: building fp32 engine ({why or 'forced'}); this takes minutes", flush=True)
        # a stale engine must not survive next to a new ready.json
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        t = time.time()
        session, ep, why = matte.make_session(model, build=True)
        if ep != "tensorrt":
            print(f"{model}: FAILED, session came up as {ep}: {why}", flush=True)
            failed += 1
            continue
        remove(Image.new("RGB", (64, 64), (128, 128, 128)), session=session)  # one real run
        built = round(time.time() - t, 1)
        engines = [{"file": os.path.basename(p), "bytes": os.path.getsize(p)}
                   for p in glob.glob(os.path.join(d, "*.engine"))]
        if not engines:
            print(f"{model}: FAILED, no .engine file written in {d}", flush=True)
            failed += 1
            continue
        with open(os.path.join(d, "ready.json"), "w", encoding="utf-8") as f:
            json.dump({"fingerprint": matte.trt_fingerprint(model), "engines": engines,
                       "build_seconds": built, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=1)
        del session
        for e in engines:
            print(f"{model}: built in {built} s -> {os.path.join(d, e['file'])} ({e['bytes'] / 1e9:.2f} GB)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    is_inside = "--inside" in args
    models = [a for a in args if not a.startswith("--")] or DEFAULT_MODELS
    sys.exit(inside(models, force) if is_inside else outer(models, force))
