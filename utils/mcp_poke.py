#!/usr/bin/env python3
"""
mcp_poke.py — quick-and-dirty CLI for testing the zeiss-inspect-api MCP server.

Speaks MCP's JSON-RPC 2.0 protocol over either stdio or Streamable HTTP.

Usage:
  # Direct corpus peek (no server involved — fastest for debugging):
  python mcp_poke.py --corpus ./corpus howtos
  python mcp_poke.py --corpus ./corpus howto scripted_elements.scripted_checks
  python mcp_poke.py --corpus ./corpus search "scripted curve check"

  # Via stdio (what LLM clients normally do):
  python mcp_poke.py --stdio "python zeiss_api_mcp.py --corpus ./corpus" \
      tool get_howto slug=scripted_elements.scripted_checks

  # Via HTTP (hits your running systemd service):
  python mcp_poke.py --http http://127.0.0.1:8765/mcp tools
  python mcp_poke.py --http http://127.0.0.1:8765/mcp \
      tool search query="scripted curve check" kind=howto

Commands (no server needed, with --corpus):
  howtos              list all indexed howto slugs (one per line)
  howto <slug>        dump one howto as JSON
  search <query>      BM25 search across all kinds
  meta                show corpus_meta.json

Commands (server mode, with --stdio or --http):
  tools               list available tool names and short descriptions
  tool <name> k=v...  call tool <name> with the given args (string values;
                      use JSON quoting for numbers/bools: limit=10 is a string
                      but limit:=10 parses as int)
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import uuid
from pathlib import Path


# ----------------------------------------------------------------------------
# Direct corpus mode (no server involved)
# ----------------------------------------------------------------------------

def cmd_direct(corpus: Path, args: list[str]) -> int:
    if not args:
        sys.exit("error: need a command. Try: howtos, howto <slug>, search <q>, meta")

    cmd, rest = args[0], args[1:]

    if cmd == "meta":
        p = corpus / "corpus_meta.json"
        if not p.exists():
            sys.exit(f"error: {p} not found. Did you rebuild with the new build_corpus.py?")
        print(p.read_text())
        return 0

    if cmd == "howtos":
        hts = json.loads((corpus / "howtos.json").read_text())
        for slug in sorted(hts):
            print(slug)
        return 0

    if cmd == "howto":
        if not rest:
            sys.exit("error: usage: howto <slug>")
        slug = rest[0]
        hts = json.loads((corpus / "howtos.json").read_text())
        if slug in hts:
            print(json.dumps(hts[slug], indent=2))
            return 0
        # try normalized
        norm = slug.replace("/", ".").replace("\\", ".").strip(".").lower()
        matches = [real for real in hts if real.lower() == norm]
        if matches:
            print(json.dumps(hts[matches[0]], indent=2))
            return 0
        # substring
        matches = sorted(s for s in hts if norm in s.lower())
        if matches:
            print(f"no exact match for {slug!r}. Candidates:", file=sys.stderr)
            for m in matches:
                print(f"  {m}", file=sys.stderr)
            return 1
        sys.exit(f"error: no howto matching {slug!r}")

    if cmd == "search":
        if not rest:
            sys.exit("error: usage: search <query>")
        # naive substring search across all indexes, enough for sanity checks
        query = " ".join(rest).lower()
        for name in ("api_functions.json", "api_classes.json", "modules.json",
                     "examples.json", "howtos.json"):
            p = corpus / name
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            hits = [k for k in data if query in k.lower()]
            if hits:
                print(f"[{name}] {len(hits)} key-substring hits:")
                for h in sorted(hits)[:20]:
                    print(f"  {h}")
        return 0

    sys.exit(f"error: unknown direct command: {cmd}")


# ----------------------------------------------------------------------------
# Server mode helpers: JSON-RPC 2.0 message framing
# ----------------------------------------------------------------------------

def _parse_kv(pairs: list[str]) -> dict:
    """Parse k=v and k:=v pairs. k:=v parses the value as JSON (for ints/bools)."""
    out: dict = {}
    for p in pairs:
        if ":=" in p:
            k, v = p.split(":=", 1)
            out[k] = json.loads(v)
        elif "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
        else:
            sys.exit(f"error: bad arg {p!r}; expected k=v or k:=v (JSON)")
    return out


def _rpc(method: str, params: dict | None = None, notify: bool = False) -> dict:
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if not notify:
        msg["id"] = uuid.uuid4().hex[:8]
    return msg


# ----------------------------------------------------------------------------
# Stdio transport
# ----------------------------------------------------------------------------

def run_stdio(cmdline: str, requests: list[dict]) -> list[dict]:
    """Launch the server as a subprocess, exchange JSON-RPC messages, return responses."""
    proc = subprocess.Popen(
        shlex.split(cmdline),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
        text=True, bufsize=1,
    )
    responses: list[dict] = []
    try:
        for req in requests:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            if "id" not in req:  # notification, no response expected
                continue
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("server closed stdout unexpectedly")
            responses.append(json.loads(line))
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=5)
    return responses


def cmd_stdio(cmdline: str, args: list[str]) -> int:
    if not args:
        sys.exit("error: need a command. Try: tools, tool <name> k=v...")

    # Standard MCP handshake.
    init = _rpc("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "mcp_poke", "version": "0.1"},
    })
    initialized = _rpc("notifications/initialized", {}, notify=True)

    if args[0] == "tools":
        reqs = [init, initialized, _rpc("tools/list")]
        resps = run_stdio(cmdline, reqs)
        tools = resps[-1].get("result", {}).get("tools", [])
        for t in tools:
            desc = (t.get("description") or "").split("\n")[0]
            print(f"  {t['name']:20s}  {desc}")
        return 0

    if args[0] == "tool":
        if len(args) < 2:
            sys.exit("error: usage: tool <name> [k=v ...]")
        name = args[1]
        call_args = _parse_kv(args[2:])
        reqs = [init, initialized, _rpc("tools/call", {"name": name, "arguments": call_args})]
        resps = run_stdio(cmdline, reqs)
        final = resps[-1]
        # Unwrap the standard MCP result envelope.
        result = final.get("result", {})
        for block in result.get("content", []):
            if block.get("type") == "text":
                # The FastMCP server serializes return dicts as JSON in a text block.
                try:
                    print(json.dumps(json.loads(block["text"]), indent=2))
                except Exception:
                    print(block["text"])
        if "error" in final:
            print(json.dumps(final["error"], indent=2), file=sys.stderr)
            return 1
        return 0

    sys.exit(f"error: unknown stdio command: {args[0]}")


# ----------------------------------------------------------------------------
# HTTP transport (Streamable HTTP)
# ----------------------------------------------------------------------------

def cmd_http(url: str, args: list[str]) -> int:
    try:
        import httpx
    except ImportError:
        sys.exit("error: pip install httpx (required for --http mode)")

    if not args:
        sys.exit("error: need a command. Try: tools, tool <name> k=v...")

    # For Streamable HTTP we POST each request and read the response.
    # The server returns either JSON (if accept=application/json) or SSE.
    # Asking for JSON-only keeps this simple.
    headers = {"Accept": "application/json, text/event-stream",
               "Content-Type": "application/json"}

    def _call(req: dict) -> dict | None:
        r = httpx.post(url, json=req, headers=headers, timeout=30.0)
        r.raise_for_status()
        if not r.text.strip():
            return None
        # Streamable HTTP may respond with SSE framing (`event:`/`data:` lines)
        # or raw JSON. Handle both.
        if r.headers.get("content-type", "").startswith("text/event-stream"):
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:].strip())
            return None
        return r.json()

    # Handshake
    _call(_rpc("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "mcp_poke", "version": "0.1"},
    }))
    _call(_rpc("notifications/initialized", {}, notify=True))

    if args[0] == "tools":
        resp = _call(_rpc("tools/list"))
        for t in resp.get("result", {}).get("tools", []):
            desc = (t.get("description") or "").split("\n")[0]
            print(f"  {t['name']:20s}  {desc}")
        return 0

    if args[0] == "tool":
        if len(args) < 2:
            sys.exit("error: usage: tool <name> [k=v ...]")
        name = args[1]
        call_args = _parse_kv(args[2:])
        resp = _call(_rpc("tools/call", {"name": name, "arguments": call_args}))
        result = (resp or {}).get("result", {})
        for block in result.get("content", []):
            if block.get("type") == "text":
                try:
                    print(json.dumps(json.loads(block["text"]), indent=2))
                except Exception:
                    print(block["text"])
        if "error" in (resp or {}):
            print(json.dumps(resp["error"], indent=2), file=sys.stderr)
            return 1
        return 0

    sys.exit(f"error: unknown http command: {args[0]}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--corpus", type=Path,
                   help="direct corpus mode (read *.json from this dir)")
    g.add_argument("--stdio", metavar="CMDLINE",
                   help="launch the server via stdio, e.g. "
                        "'python zeiss_api_mcp.py --corpus ./corpus'")
    g.add_argument("--http", metavar="URL",
                   help="talk to a running HTTP server, e.g. "
                        "'http://127.0.0.1:8765/mcp'")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="command and args")
    args = ap.parse_args()

    if args.corpus:
        return cmd_direct(args.corpus, args.rest)
    if args.stdio:
        return cmd_stdio(args.stdio, args.rest)
    if args.http:
        return cmd_http(args.http, args.rest)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
