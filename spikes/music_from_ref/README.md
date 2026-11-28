# music_from_ref spike (2026-11-28)

"New music" today = a text prompt the manager writes; nothing measured from the reference reaches
the generator, so the bed's tempo, key and lift land anywhere. This spike tests beds that take the
reference's *style and clock* but not its tune. Four farm python jobs, in order:

1. `analyze.py prep jobs/r1.json out`   reference stems (htdemucs) + tempo/beats/key/lift
2. `ace_job.py jobs/r1.json out`        ACE-Step 1.5 (authors' uv install): understand -> B1 style-ref, B2 cover 0.3
3. `yue_job.py jobs/r1.json out`        YuE2: A conformed new score, A2 chord skeleton, X melody copy (control)
   (venv = ../yue2_official/requirements.txt)
4. `analyze.py measure jobs/r1.json out` every take vs its reference + mp3s for the rating page

Jobs 1, 2 and 4 use `requirements.txt` here. State lives on the PC in ~/music-from-ref/.
Arm C = the bed the real job got (generate_music from the manager's prompt).
