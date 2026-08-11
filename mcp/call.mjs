// Drives the MCP server over stdio exactly like Claude Code does, loading
// credentials from ../.env — useful for testing changes without restarting
// Claude Code (whose MCP client only picks up new tools on restart).
//
//   node call.mjs                       # list tools + check new params exist
//   node call.mjs list_jobs '{"limit":3}'
//   node call.mjs sync_assets '{"paths":["C:/path/to/clips"]}'
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const env = { ...process.env };
for (const line of fs.readFileSync(path.join(HERE, "..", ".env"), "utf8").split(/\r?\n/)) {
  const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/);
  if (m) env[m[1]] = m[2].replace(/^["']|["']$/g, "");
}

const transport = new StdioClientTransport({
  command: process.execPath,
  args: [path.join(HERE, "index.js")],
  env,
});
const client = new Client({ name: "e2e", version: "1.0.0" });
await client.connect(transport);

const [cmd, argJson] = process.argv.slice(2);

if (!cmd || cmd === "tools") {
  const tools = await client.listTools();
  console.log("TOOLS:", tools.tools.map((t) => t.name).join(", "));
  const sync = tools.tools.find((t) => t.name === "sync_assets");
  console.log("sync_assets schema:", JSON.stringify(sync?.inputSchema ?? null));
  const sub = tools.tools.find((t) => t.name === "submit_render_job");
  console.log("submit has quality:", !!sub?.inputSchema?.properties?.quality,
    "| has assets:", !!sub?.inputSchema?.properties?.assets);
} else {
  const res = await client.callTool({ name: cmd, arguments: argJson ? JSON.parse(argJson) : {} });
  console.log(`[${cmd}] isError=${!!res.isError}`);
  console.log(res.content.map((c) => c.text).join("\n"));
}

await client.close();
