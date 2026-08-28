# Odyssey subtitle cleanup

GPU video-inpainting job for the approved Honda Odyssey source video.

- ProPainter fills the fixed subtitle band and the one-second opening title.
- The original AAC audio is preserved.
- The cleaned picture is exported as 1080x1920 H.264.
- `--draft-seconds 4` creates a short validation clip before the full job.
