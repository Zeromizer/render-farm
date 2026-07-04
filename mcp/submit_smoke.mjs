// Submit a job through the MCP server itself (tests the submit path end-to-end).
// Usage: node submit_smoke.mjs '<json args for submit_render_job>'
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const args = JSON.parse(process.argv[2]);
const transport = new StdioClientTransport({
  command: process.execPath,
  args: [new URL("./index.js", import.meta.url).pathname.slice(1)],
  env: { ...process.env },
});
const client = new Client({ name: "submit-smoke", version: "1.0.0" });
await client.connect(transport);
const res = await client.callTool({ name: "submit_render_job", arguments: args });
console.log(res.content[0].text);
await client.close();
