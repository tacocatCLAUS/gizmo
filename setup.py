import json
import subprocess
import os
import time
import sys

CONFIG_PATH = "config.json"
STARTUP_TIMEOUT = 3  # seconds to wait for a response

def start_mcp_server(server_conf):
    env = os.environ.copy()
    if "env" in server_conf:
        env.update(server_conf["env"])

    return subprocess.Popen(
        [server_conf["command"]] + server_conf.get("args", []),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

def send_msg(proc, msg):
    raw = json.dumps(msg)
    header = f"Content-Length: {len(raw)}\r\n\r\n"
    try:
        proc.stdin.write(header + raw)
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        return False
    return True

def read_msg(proc, timeout=STARTUP_TIMEOUT):
    """Read MCP JSON-RPC message with a timeout. Works on Windows & Unix."""
    start = time.time()
    headers = {}
    length = None

    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        if line in ("\n", "\r\n"):
            # End of headers
            if length is None:
                length = int(headers.get("Content-Length", 0))
            if length:
                body = proc.stdout.read(length)
                if body:
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError:
                        return None
            break
        else:
            if ":" in line:
                name, value = line.strip().split(":", 1)
                headers[name.strip()] = value.strip()

    return None  # timed out or invalid message

def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    for name, server_conf in config.get("mcpServers", {}).items():
        print(f"\n--- {name} ---")
        try:
            proc = start_mcp_server(server_conf)
        except Exception as e:
            print(f"❌ Failed to start: {e}")
            continue

        if not send_msg(proc, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }):
            print("❌ Could not send initialize request")
            proc.terminate()
            continue

        init_resp = read_msg(proc)
        if not init_resp:
            print("❌ No initialize response")
            proc.terminate()
            continue

        send_msg(proc, {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        tools_resp = read_msg(proc)
        if tools_resp and "result" in tools_resp:
            tools = tools_resp["result"].get("tools", [])
            if tools:
                for tool in tools:
                    print(f"Tool: {tool['name']}")
                    print(f"  Description: {tool['description']}")
            else:
                print("ℹ No tools found")
        else:
            print("❌ Could not retrieve tools list")

        proc.terminate()

if __name__ == "__main__":
    main()
