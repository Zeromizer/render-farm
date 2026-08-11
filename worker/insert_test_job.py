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
    ap.add_argument("--engine", required=True, choices=["remotion", "blender"])
    ap.add_argument("--repo", required=True)
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

    row = db.sb.table("farm_render_jobs").insert({
        "status": "pending",
        "engine": args.engine,
        "repo_url": args.repo,
        "git_ref": args.ref,
        "params": params,
    }).execute().data[0]
    print(f"inserted job {row['id']}")


if __name__ == "__main__":
    main()
