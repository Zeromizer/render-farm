"""Insert a test render job directly (PC-side smoke test, bypasses the MCP).

Examples:
  python insert_test_job.py --engine remotion --repo https://github.com/user/proj --composition Demo
  python insert_test_job.py --engine blender --repo https://github.com/user/proj \
      --blend-file scenes/test.blend --frame-start 1 --frame-end 24
"""
import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import db  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["remotion", "blender", "video_gen", "aftereffects"])
    ap.add_argument("--repo", help="git URL (not used by video_gen)")
    ap.add_argument("--params", help="video_gen: JSON for params.video_gen (prompt, mode, duration_s, ...); "
                                     "aftereffects: JSON for params.aftereffects (or @file.json)")
    ap.add_argument("--timeout-minutes", type=int)
    ap.add_argument("--priority", type=int)
    ap.add_argument("--ref", default="main")
    ap.add_argument("--composition")
    ap.add_argument("--project-dir")
    ap.add_argument("--entry")
    ap.add_argument("--codec")
    ap.add_argument("--frame-range")
    ap.add_argument("--props", help="JSON string")
    ap.add_argument("--quality", choices=["draft", "final"])
    ap.add_argument("--assets", help="JSON string: [{path, sha256, size}, ...] (from sync_assets)")
    ap.add_argument("--blend-file")
    ap.add_argument("--frame-start", type=int)
    ap.add_argument("--frame-end", type=int)
    ap.add_argument("--single-frame", type=int)
    ap.add_argument("--output-format")
    args = ap.parse_args()

    params = {}
    for key in ("composition", "project_dir", "entry", "codec", "frame_range", "quality",
                "blend_file", "frame_start", "frame_end", "single_frame", "output_format"):
        v = getattr(args, key)
        if v is not None:
            params[key] = v
    if args.props:
        params["props"] = json.loads(args.props)
    if args.assets:
        params["assets"] = json.loads(args.assets)
    if args.engine == "video_gen":
        if not args.params:
            ap.error("video_gen needs --params '<json>' with at least a prompt")
        params = {"video_gen": json.loads(args.params)}
    elif args.engine == "aftereffects":
        if not args.params:
            ap.error("aftereffects needs --params '<json>' or @file.json (params.aftereffects)")
        raw = open(args.params[1:], encoding="utf-8").read() if args.params.startswith("@") else args.params
        params = {"aftereffects": json.loads(raw)}
    elif not args.repo:
        ap.error("--repo is required for this engine")

    row = {
        "status": "pending",
        "engine": args.engine,
        "repo_url": args.repo or "-",
        "git_ref": args.ref,
        "params": params,
    }
    if args.timeout_minutes:
        row["timeout_minutes"] = args.timeout_minutes
    if args.priority is not None:
        row["priority"] = args.priority
    row = db.sb.table("farm_render_jobs").insert(row).execute().data[0]
    print(f"inserted job {row['id']}")


if __name__ == "__main__":
    main()
