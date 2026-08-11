// Stdio smoke test for the render-farm MCP server: spawns it, lists tools,
// calls list_jobs and get_job_status against the real Supabase queue.
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const JOB_ID = process.argv[2];

const transport = new StdioClientTransport({
  command: process.execPath,
  args: [new URL("./index.js", import.meta.url).pathname.replace(/^\//, "")],
  env: { ...process.env },
});
const client = new Client({ name: "smoke", version: "1.0.0" });
await client.connect(transport);

const tools = await client.listTools();
console.log("TOOLS:", tools.tools.map((t) => t.name).join(", "));

const list = await client.callTool({ name: "list_jobs", arguments: { limit: 3 } });
console.log("LIST_JOBS:", list.content[0].text.slice(0, 400));

if (JOB_ID) {
  const st = await client.callTool({ name: "get_job_status", arguments: { job_id: JOB_ID } });
  console.log("STATUS:", st.content[0].text);
  const dl = await client.callTool({
    name: "download_result",
    arguments: { job_id: JOB_ID, dest_path: "C:/Users/SHAWN_~1/AppData/Local/Temp/claude/c--Coding-Voice-Output/f13f13ee-2ee6-4da7-8cdf-15712c1b8749/scratchpad/mcp_downloaded.mp4" },
  });
  console.log("DOWNLOAD:", dl.content[0].text);
}

await client.close();
