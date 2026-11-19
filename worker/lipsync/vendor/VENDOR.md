# Vendored: joonson/syncnet_python

- Upstream: https://github.com/joonson/syncnet_python
- Commit: 907c0b579c2e2d83f0eae1b2ac9e720cde4e5623 (2026-04-17)
- Files copied unmodified: SyncNetInstance.py, SyncNetModel.py, detectors/__init__.py,
  detectors/s3fd/{__init__,box_utils,nets}.py, LICENSE.md (MIT, code only).
- Not copied: run_pipeline.py / run_syncnet.py (scripts with argparse at import time). The pieces
  we use from run_pipeline.py (bb_intersection_over_union, track_shot, crop_video) are copied into
  ../lipsync_check.py with that note.
- Weights are NOT vendored. lipsync_check.py downloads them from the URLs in upstream's
  download_model.sh into the worker cache and checks sha256:
  - syncnet_v2.model  961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442
  - sfd_face.pth      d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491
- Weights licence: the VGG download page offers them for research use (CC-BY wording, cite
  Chung & Zisserman 2016); no explicit commercial grant. We use them only as an internal QA
  measure on our own renders (nothing they produce ships). If that is not acceptable, swap in
  LatentSync's StableSyncNet (ByteDance, Apache-2.0 code) and recalibrate the thresholds.
- Reproduced upstream's README demo on 2026-11-19 (laptop CPU, torch 2.5.1):
  demo_syncnet.py example.avi -> AV offset 3, min dist 5.358, confidence 10.081
  (README: 3 / 5.353 / 10.021); full pipeline run_pipeline + run_syncnet -> 3 / 6.564 / 8.340,
  which lipsync_check.py matches exactly.
