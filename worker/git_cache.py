"""Per-repo clone cache: clone once, fetch+checkout per job, npm ci only when
the lockfile changed. Short hashed dir names keep paths under MAX_PATH."""
import hashlib
import os
import shutil
import subprocess
import time

import config

LOCKHASH_FILE = ".render-farm-lockhash"


def _run_git(args, cwd=None):
    res = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()[:800]}")
    return res.stdout


def repo_dir(repo_url):
    h = hashlib.sha1(repo_url.encode()).hexdigest()[:12]
    return os.path.join(config.REPOS_DIR, h)


def rmtree_robust(path):
    """rmtree that survives node_modules long paths (\\\\?\\ prefix on Windows)
    and git's read-only object files (chmod +w on failure, then retry)."""
    import stat

    def _on_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    target = path
    if os.name == "nt":
        target = "\\\\?\\" + os.path.abspath(path)
    shutil.rmtree(target, onerror=_on_error)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    return not os.path.exists(path)


def _clone_fresh(repo_url, d, log):
    log(f"cloning {repo_url}")
    _run_git(["clone", repo_url, d])
    _run_git(["config", "core.longpaths", "true"], cwd=d)


def checkout(repo_url, ref, log):
    """Clone-or-fetch repo_url and force-checkout ref. Returns the repo dir.

    A cached dir that fails any git op (e.g. left half-deleted by an
    interrupted cleanup) is wiped and re-cloned once.
    """
    d = repo_dir(repo_url)
    if not os.path.isdir(os.path.join(d, ".git")):
        if os.path.exists(d):  # remnant without .git — a corrupt cache
            rmtree_robust(d)
        _clone_fresh(repo_url, d, log)
    else:
        try:
            log(f"fetching {repo_url}")
            _run_git(["fetch", "--all", "--prune", "--tags"], cwd=d)
        except RuntimeError as e:
            log(f"cached repo unusable ({str(e)[:120]}), re-cloning")
            if not rmtree_robust(d):
                raise RuntimeError(f"cannot remove corrupt repo cache {d}")
            _clone_fresh(repo_url, d, log)

    # Prefer the remote tip when ref is a branch name; fall back to raw ref (sha/tag).
    try:
        _run_git(["checkout", "--force", f"origin/{ref}"], cwd=d)
    except RuntimeError:
        _run_git(["checkout", "--force", ref], cwd=d)
    _run_git(["clean", "-fd", "-e", "node_modules", "-e", LOCKHASH_FILE], cwd=d)
    return d


def _file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def npm_ci_if_needed(project_dir, log, run_streaming_kwargs):
    """Run npm ci when node_modules is missing or package-lock.json changed."""
    lock = os.path.join(project_dir, "package-lock.json")
    marker = os.path.join(project_dir, LOCKHASH_FILE)
    if not os.path.exists(lock):
        raise RuntimeError("no package-lock.json in project dir — commit one (npm install locally first)")
    want = _file_hash(lock)
    have = None
    if os.path.exists(marker):
        with open(marker) as f:
            have = f.read().strip()
    if want == have and os.path.isdir(os.path.join(project_dir, "node_modules")):
        log("node_modules up to date, skipping npm ci")
        return
    log("running npm ci (may take several minutes)")
    from proc import run_streaming

    npm = shutil.which("npm.cmd") or shutil.which("npm")
    run_streaming([npm, "ci", "--no-audit", "--no-fund"], cwd=project_dir, **run_streaming_kwargs)
    with open(marker, "w") as f:
        f.write(want)


def cleanup_old(log):
    """Delete repo caches untouched >14 days and work dirs >2 days."""
    now = time.time()
    for base, max_days in ((config.REPOS_DIR, config.REPO_CACHE_MAX_AGE_DAYS),
                           (config.WORK_DIR, config.WORK_MAX_AGE_DAYS)):
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            p = os.path.join(base, name)
            try:
                if now - os.path.getmtime(p) > max_days * 86400:
                    log(f"cleanup: removing {p}")
                    if not rmtree_robust(p):
                        log(f"cleanup: {p} only partially removed (locked files?)")
            except OSError:
                pass
