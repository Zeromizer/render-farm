"""API-format ComfyUI graph that tracks a named kind of object through a clip with SAM 3.1
(native SAM3 nodes, ComfyUI >= 0.37), for clip_edit: move mode finds the car being moved in
the source and again in each render.

  CheckpointLoaderSimple(sam3.1) -> CLIPTextEncode(noun)
  LoadVideo -> GetVideoComponents -> SAM3_Detect(first frame) -> SAM3_VideoTrack
  (re-detects every few frames, so objects that enter later are picked up)
  -> SAM3_TrackToMask per object index -> MaskToImage -> CreateVideo -> SaveVideo

It tracks EVERY instance of the noun ("car" finds the hero car too); clip_edit keeps the
object under the hint point. One mask video per object index, named <prefix>_obj<k>.
"""
CKPT = "sam3.1_multiplex_fp16.safetensors"
REQUIRED_NODES = ("SAM3_Detect", "SAM3_VideoTrack", "SAM3_TrackToMask", "CheckpointLoaderSimple", "LoadVideo",
                  "GetVideoComponents", "ImageFromBatch", "MaskToImage", "CreateVideo", "SaveVideo")


def build(src_name, noun, prefix, objects=6, fps=24.0, detect_interval=4, threshold=0.4, track_threshold=0.5):
    """Return the API prompt dict for tracking up to `objects` instances of `noun`."""
    if not (noun or "").strip():
        raise ValueError("object tracking needs a noun")
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoadVideo", "inputs": {"file": src_name}},
        "3": {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": noun, "clip": ["1", 1]}},
        "5": {"class_type": "ImageFromBatch", "inputs": {"image": ["3", 0], "batch_index": 0, "length": 1}},
        "6": {"class_type": "SAM3_Detect", "inputs": {"model": ["1", 0], "image": ["5", 0], "conditioning": ["4", 0],
                                                      "threshold": float(threshold), "refine_iterations": 2,
                                                      "individual_masks": False}},
        "7": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["3", 0], "model": ["1", 0], "initial_mask": ["6", 0],
                                                          "conditioning": ["4", 0],
                                                          "detection_threshold": float(track_threshold),
                                                          "max_objects": int(objects),
                                                          "detect_interval": int(detect_interval)}},
    }
    for k in range(int(objects)):
        b = 100 + 10 * k
        g[str(b)] = {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["7", 0], "object_indices": str(k)}}
        g[str(b + 1)] = {"class_type": "MaskToImage", "inputs": {"mask": [str(b), 0]}}
        g[str(b + 2)] = {"class_type": "CreateVideo", "inputs": {"images": [str(b + 1), 0], "fps": float(fps)}}
        g[str(b + 3)] = {"class_type": "SaveVideo", "inputs": {"video": [str(b + 2), 0], "filename_prefix": f"{prefix}_obj{k}",
                                                              "format": "mp4", "codec": "h264"}}
    return g
