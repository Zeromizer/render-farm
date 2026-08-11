"""One-shot setup: create/update the private 'renders' and 'assets' Storage buckets.

The farm_render_jobs table + RPCs are DDL and must be applied via the dashboard
SQL editor (see schema.sql) -- PostgREST cannot run DDL.

Note: the per-bucket file_size_limit is subordinate to the project-global upload
cap (Dashboard -> Storage -> Settings; 50 MB on the Free plan). Raise that too
if large assets/outputs still 413.
"""
import os

from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

BUCKETS = ("renders", "assets")
OPTIONS = {"public": False, "file_size_limit": "2GB"}

existing = [b.name for b in sb.storage.list_buckets()]
for bucket in BUCKETS:
    if bucket in existing:
        sb.storage.update_bucket(bucket, options=OPTIONS)
        print(f"bucket '{bucket}' already exists — updated file_size_limit to 2GB")
    else:
        sb.storage.create_bucket(bucket, options=OPTIONS)
        print(f"created private bucket '{bucket}' (file_size_limit 2GB)")

print("reminder: the project-global upload cap (Storage settings) still applies "
      "on top of these per-bucket limits")
