import { createClient } from "@supabase/supabase-js";

const url = process.env.SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_KEY;
if (!url || !key) {
  console.error("SUPABASE_URL and SUPABASE_SERVICE_KEY env vars are required");
  process.exit(1);
}

export const sb = createClient(url, key);
export const BUCKET = "renders";
