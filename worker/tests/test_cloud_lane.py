"""cloud_lane on RunPod REST v2 (v1 was retired in Nov 2026). Run from worker/
with the worker venv: python -m unittest tests.test_cloud_lane"""
import os
import sys
import unittest
from unittest import mock

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)

import cloud_lane as CL  # noqa: E402  (needs the worker venv: db -> supabase)


class Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text or str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class ApiTest(unittest.TestCase):
    def test_base_url_is_v2(self):
        self.assertEqual(CL.API, "https://api.runpod.io/v2")

    def test_list_pods_unwraps_and_follows_cursor(self):
        pages = [
            Resp(200, {"pods": [{"id": "a", "name": "rf-cloud-1"}, {"id": "x", "name": "other"}],
                       "pagination": {"hasNextPage": True, "nextCursor": "c1"}}),
            Resp(200, {"pods": [{"id": "b", "name": "rf-cloud-2"}],
                       "pagination": {"hasNextPage": False, "nextCursor": None}}),
        ]
        with mock.patch.object(CL.httpx, "get", side_effect=pages) as get:
            pods = CL.list_pods()
        self.assertEqual([p["id"] for p in pods], ["a", "b"])
        self.assertEqual(get.call_args_list[1].kwargs["params"], {"cursor": "c1"})
        self.assertTrue(get.call_args_list[0].args[0].startswith("https://api.runpod.io/v2/pods"))

    def test_delete_404_is_already_gone(self):
        with mock.patch.object(CL.httpx, "delete", return_value=Resp(404)):
            CL.delete_pod("p1")  # no raise
        with mock.patch.object(CL.httpx, "delete", return_value=Resp(500)):
            with self.assertRaises(RuntimeError):
                CL.delete_pod("p1")

    def test_body_has_v2_shape(self):
        b = CL.pod_body({"remake", "video_gen"}, "NVIDIA GeForce RTX 5090")
        self.assertEqual(b["gpu"], {"id": "NVIDIA GeForce RTX 5090", "count": 1})
        self.assertEqual(b["mounts"]["network"][0]["path"], "/workspace")
        self.assertEqual(b["cmd"][:2], ["bash", "-c"])
        self.assertEqual(b["env"]["WORKER_ENGINES"], "remake,video_gen")
        for v1 in ("imageName", "cloudType", "gpuTypeIds", "gpuCount", "networkVolumeId",
                   "volumeMountPath", "containerDiskInGb", "supportPublicIp", "dockerStartCmd"):
            self.assertNotIn(v1, b)  # v2 answers 422 to any unknown field

    def test_create_walks_gpu_types_in_order(self):
        calls = []

        def post(url, headers, json, timeout):
            calls.append(json["gpu"]["id"])
            if json["gpu"]["id"] == "5090":
                return Resp(400, text="no longer any instances available")
            return Resp(201, {"id": "pod_9", "gpu": {"id": json["gpu"]["id"]}, "cost": 2.69})

        with mock.patch.object(CL.httpx, "post", side_effect=post):
            pod = CL.create_pod({"remake"}, ["5090", "PRO6000"])
        self.assertEqual(calls, ["5090", "PRO6000"])
        self.assertEqual(CL.pod_gpu(pod), "PRO6000")

    def test_create_stops_on_422_and_reports_every_error(self):
        with mock.patch.object(CL.httpx, "post", return_value=Resp(422, text="bad field")) as post:
            with self.assertRaises(RuntimeError) as e:
                CL.create_pod({"remake"}, ["5090", "PRO6000"])
        self.assertEqual(post.call_count, 1)
        self.assertIn("5090: RunPod 422", str(e.exception))


class PodWorkTest(unittest.TestCase):
    def test_matches_id_with_and_without_pod_prefix(self):
        q = mock.MagicMock()
        q.select.return_value = q
        q.in_.return_value = q
        q.order.return_value = q
        q.limit.return_value = q
        q.execute.return_value = mock.Mock(data=[{"status": "processing"}])
        with mock.patch.object(CL.db, "sb", mock.Mock(table=mock.Mock(return_value=q))):
            busy, _ = CL._pod_work("pod_abc")
        self.assertTrue(busy)
        self.assertEqual(q.in_.call_args.args, ("claimed_by", ["cloud:abc", "cloud:pod_abc"]))


if __name__ == "__main__":
    unittest.main()
