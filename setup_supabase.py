"""One-shot setup: create the private 'renders' Storage bucket.

The render_jobs table + RPCs are DDL and must be applied via the dashboard
SQL editor (see schema.sql) -- PostgREST cannot run DDL.
"""
import os

from dotenv import load_dotenv
from supabase import create_client

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

BUCKET = "renders"

existing = [b.name for b in sb.storage.list_buckets()]
if BUCKET in existing:
    print(f"bucket '{BUCKET}' already exists")
else:
    sb.storage.create_bucket(BUCKET, options={"public": False})
    print(f"created private bucket '{BUCKET}'")
