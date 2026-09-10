"""Fidelity log parsing and summary (videogen/fidelity.py). The ffmpeg-backed
measure() is exercised only when ffmpeg/ffprobe are on PATH (skipped
otherwise) on two tiny synthetic clips."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogen import fidelity  # noqa: E402

SSIM_LOG = """n:1 Y:0.987654 U:0.990000 V:0.991000 All:0.988000 (19.2)
n:2 Y:0.900000 U:0.950000 V:0.950000 All:0.910000 (10.5)
n:3 Y:0.700000 U:0.800000 V:0.800000 All:0.720000 (5.5)
"""
PSNR_LOG = """n:1 mse_avg:0.50 mse_y:0.6 mse_u:0.3 mse_v:0.3 psnr_avg:51.14 psnr_y:50.3 psnr_u:53.4 psnr_v:53.4
n:2 mse_avg:0.00 mse_y:0.0 mse_u:0.0 mse_v:0.0 psnr_avg:inf psnr_y:inf psnr_u:inf psnr_v:inf
n:3 mse_avg:120.0 mse_y:130 mse_u:100 mse_v:100 psnr_avg:27.34 psnr_y:26.9 psnr_u:28.1 psnr_v:28.1
"""


class Parsing(unittest.TestCase):
    def test_ssim(self):
        rows = fidelity.parse_ssim_log(SSIM_LOG)
        self.assertEqual([r["n"] for r in rows], [1, 2, 3])
        self.assertAlmostEqual(rows[0]["All"], 0.988)

    def test_psnr_inf(self):
        rows = fidelity.parse_psnr_log(PSNR_LOG)
        self.assertEqual(rows[1]["psnr_avg"], fidelity.INF_PSNR)
        self.assertAlmostEqual(rows[2]["psnr_avg"], 27.34)


class Summary(unittest.TestCase):
    def test_flags_and_verdict(self):
        ssim = fidelity.parse_ssim_log(SSIM_LOG)
        psnr = fidelity.parse_psnr_log(PSNR_LOG)
        cells = {(r, c): [0.95, 0.9, 0.85] for r in range(4) for c in range(4)}
        cells[(1, 2)] = [0.95, 0.9, 0.60]     # one cell falls away on frame 3
        res = fidelity.summarise(ssim, psnr, cells, 4)
        self.assertEqual(res["frames"], 3)
        self.assertEqual(res["verdict"], "review")
        self.assertEqual(res["flags"]["frames_below_ssim"], [2])
        self.assertEqual(res["flags"]["frames_below_psnr"], [2])
        drift = res["flags"]["cells_drift"]
        self.assertEqual(len(drift), 1)
        self.assertEqual((drift[0]["frame"], drift[0]["row"], drift[0]["col"]), (2, 1, 2))
        self.assertEqual(res["grid"]["cell_min"][1][2], 0.6)
        self.assertAlmostEqual(res["ssim"]["min"], 0.72)
        self.assertEqual(res["thresholds"]["ssim_min"], 0.8)

    def test_clean_is_ok(self):
        ssim = [{"n": i, "Y": 0.95, "U": 0.95, "V": 0.95, "All": 0.95} for i in range(5)]
        psnr = [{"n": i, "mse_avg": 1, "psnr_avg": 40.0, "psnr_y": 40.0} for i in range(5)]
        cells = {(r, c): [0.94] * 5 for r in range(4) for c in range(4)}
        res = fidelity.summarise(ssim, psnr, cells, 4, {"ssim_min": 0.9})
        self.assertEqual(res["verdict"], "ok")
        self.assertEqual(res["flags"]["cells_drift"], [])


def _ffmpeg():
    return shutil.which("ffmpeg") and shutil.which("ffprobe")


@unittest.skipUnless(_ffmpeg(), "ffmpeg not on PATH")
class Measure(unittest.TestCase):
    def test_identical_and_blurred(self):
        tmp = tempfile.mkdtemp(prefix="fid_")
        src = os.path.join(tmp, "src.mp4")
        up = os.path.join(tmp, "up.mp4")
        blur = os.path.join(tmp, "blur.mp4")
        # segments._tool resolves via runners.hyperframes which needs config; point it at PATH here.
        from videogen import segments
        segments._BIN.update({"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")})
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=24:duration=0.5",
                        "-pix_fmt", "yuv420p", src], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "scale=256:192:flags=lanczos", "-pix_fmt", "yuv420p",
                        "-crf", "10", up], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "scale=256:192,boxblur=4", "-pix_fmt", "yuv420p",
                        "-crf", "10", blur], check=True)
        clean = fidelity.measure(src, up, os.path.join(tmp, "a"), log=lambda *_: None)
        blurred = fidelity.measure(src, blur, os.path.join(tmp, "b"), log=lambda *_: None)
        self.assertGreater(clean["ssim"]["mean"], 0.9)
        self.assertGreater(clean["ssim"]["mean"], blurred["ssim"]["mean"])
        self.assertEqual(clean["frames"], 12)
        cmp_path = fidelity.compare_video(src, up, os.path.join(tmp, "cmp.mp4"), log=lambda *_: None)
        self.assertEqual(fidelity._info(cmp_path)["width"], 512)


if __name__ == "__main__":
    unittest.main()
