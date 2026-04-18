# zi-atlas

An MCP server that exposes the ZEISS INSPECT App Python API — functions,
classes, modules, how-to guides, and full example apps — to LLM clients
through a compact tool surface with hybrid lexical + semantic search.

Two scripts:

- `src/build_corpus.py` — clones the two upstream ZEISS repos, parses the
  docs / examples / how-tos, and emits a handful of JSON files under
  `./corpus/`.
- `src/zeiss_api_mcp.py` — MCP server that loads the corpus and serves
  tool calls over stdio or streamable HTTP.

A helper CLI lives at `utils/mcp_poke.py` for poking at a running server
or the raw corpus without needing an LLM client.

---

## Prerequisites

- Python 3.11+
- `git` on PATH (needed by `build_corpus.py` to clone the upstream repos)
- Network access on first run (for the clones and — if you enable semantic
  search — for the embedding model download)

## Install

```bash
git clone https://github.com/plopperzzz/zi-atlas.git
cd zi-atlas

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Required
pip install "mcp[cli]" rank_bm25

# Optional — enables hybrid (BM25 + dense) search. Omit to run BM25-only.
pip install sentence-transformers numpy
```

`sentence-transformers` pulls in PyTorch. If you don't want that dependency,
skip it and the server will log a note and serve BM25-only; the rest of the
tool surface is unaffected.

## Step 1 — build the corpus

```bash
python src/build_corpus.py --zeiss-version 2026
```

This will clone `zeiss-inspect-app-api` and `zeiss-inspect-app-examples`
into `./repos/`, check out the `2026` branch on each, and write the parsed
corpus to `./corpus/`:

```
corpus/
  api_functions.json
  api_classes.json
  modules.json
  howtos.json
  examples.json
  corpus_meta.json
```

Useful flags (`--help` for the full list):

| flag | purpose |
| --- | --- |
| `--zeiss-version YEAR` | branch name on the api repo (default `2026`) |
| `--ex-branch BRANCH`   | override the examples-repo branch (e.g. for an unmerged feature branch) |
| `--api-repo DIR`       | use an existing clone instead of auto-cloning |
| `--ex-repo DIR`        | same for the examples repo |
| `--out DIR`            | corpus output directory (default `./corpus`) |
| `--no-fetch`           | skip `git fetch` / `pull` (offline mode) |

## Step 2 — run the MCP server

### stdio (for local LLM clients)

```bash
python src/zeiss_api_mcp.py --corpus ./corpus
```

This is the transport Claude Desktop / Claude Code / most IDE integrations
use. Point your client at the command above.

### streamable HTTP (for remote / systemd deployment)

```bash
python src/zeiss_api_mcp.py --corpus ./corpus --http 127.0.0.1:8765
```

If a web UI fronts the server (Open WebUI, etc.) you'll need to whitelist
its origin and host:

```bash
python src/zeiss_api_mcp.py --corpus ./corpus \
    --http 0.0.0.0:8765 \
    --allow-origin http://192.168.1.22:8082 \
    --allow-host 192.168.1.22:8765
```

For a trusted LAN you can bypass the DNS-rebinding guard entirely with
`--no-security`. Don't do that on anything exposed to the internet.

### Semantic-search flags

The first run encodes every function, class, module, example, and how-to
with the embedding model and caches the float32 vectors under
`<corpus-dir>/.embeddings/`. Subsequent starts are fast — a kind is only
re-encoded when its content signature changes.

| flag | default | purpose |
| --- | --- | --- |
| `--embedding-model NAME` | `BAAI/bge-small-en-v1.5` | any sentence-transformers model id. `bge-small` (~33M params, 384-dim) runs fine on CPU; switch to `nomic-ai/nomic-embed-text-v1.5` for higher quality if you have GPU headroom. |
| `--embedding-cache-dir DIR` | `<corpus>/.embeddings` | where to store cached `.npz` vectors |
| `--rebuild-embeddings` | off | ignore cache and recompute everything |
| `--embedding-device DEV` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` \| `mps`. Auto probes the GPU's compute capability and falls back to CPU if it isn't supported by the installed PyTorch wheel. |
| `--no-semantic` | off | disable dense search; serve BM25 only |

### "CUDA error: no kernel image is available for execution on the device"

The default PyTorch wheel only supports modern CUDA archs (sm_70+). Older
cards — Tesla M40 (sm_52), Tesla K80 (sm_37), GTX 9xx (sm_52), etc. —
report `cuda.is_available() == True` but blow up on the first kernel
launch. With `--embedding-device auto` the server detects this up front
and switches to CPU. If you want to force CPU explicitly:

```bash
python src/zeiss_api_mcp.py --corpus ./corpus --embedding-device cpu
```

For the M40 specifically, encoding the full corpus on CPU takes ~30-60s
on first run; subsequent starts use the cache and are instant.

## Tool surface

| tool | what it does |
| --- | --- |
| `lookup_function(name)` | signature, params, return info, cross-refs. Accepts full fqn or short name. |
| `lookup_class(name)`    | class description, methods, cross-refs. |
| `lookup_module(name)`   | module description + function / class listing. |
| `dump_module(name, include_extended=False)` | every function and class in a module with full descriptions — use instead of paginating through search. |
| `list_all_symbols(prefix="", kind="all", limit=1000)` | flat enumeration of every documented symbol. Optional case-insensitive substring filter. Good for "what exists?" exploration. |
| `get_example(name, full_code=True)` | full example doc, all scripts, and the list of `gom.api.*` calls made. |
| `get_howto(slug)` | full how-to guide content. Accepts dots or slashes. |
| `search(query, kind="all", limit=25, mode="hybrid")` | hybrid BM25 + dense semantic search, fused with RRF. `mode` is `"hybrid"` (default), `"bm25"`, or `"semantic"`. Falls back to BM25 if `sentence-transformers` isn't installed. |
| `search_by_tag(tag)` | examples by tag (derived from category + name tokens when upstream tags are empty). |
| `list_modules()` | all modules with function / class counts. |

`search` returns `mode` and `semantic_available` alongside the hits so the
client can see whether it actually ran in hybrid mode.

## Client configuration examples

### Claude Desktop / Claude Code

Add to `claude_desktop_config.json` (or your project's MCP config):

```json
{
  "mcpServers": {
    "zeiss-inspect-api": {
      "command": "python",
      "args": [
        "/absolute/path/to/zi-atlas/src/zeiss_api_mcp.py",
        "--corpus",
        "/absolute/path/to/zi-atlas/corpus"
      ]
    }
  }
}
```

If you're using a virtualenv, point `command` at its Python, e.g.
`/absolute/path/to/zi-atlas/.venv/bin/python`.

### systemd (HTTP mode)

```ini
[Unit]
Description=ZEISS INSPECT API MCP server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/zi-atlas
ExecStart=/opt/zi-atlas/.venv/bin/python src/zeiss_api_mcp.py \
    --corpus /opt/zi-atlas/corpus \
    --http 127.0.0.1:8765
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Poking at it without a client

`utils/mcp_poke.py` speaks the MCP JSON-RPC directly and also has a
"direct corpus" mode that skips the server entirely:

```bash
# Direct — hits the JSON files, no server:
python utils/mcp_poke.py --corpus ./corpus howtos
python utils/mcp_poke.py --corpus ./corpus search "scripted curve check"

# stdio:
python utils/mcp_poke.py \
    --stdio "python src/zeiss_api_mcp.py --corpus ./corpus" \
    tool search query="how do I transform coordinates" kind=function

# HTTP:
python utils/mcp_poke.py --http http://127.0.0.1:8765/mcp tools
python utils/mcp_poke.py --http http://127.0.0.1:8765/mcp \
    tool search query="camera projection" mode=semantic
```

## Refreshing the corpus

Rerun `build_corpus.py` with the same `--zeiss-version`. It will `git
fetch` the upstream repos and re-parse. The embedding cache is keyed by
content signature, so only changed kinds get re-encoded at next server
start.

## Layout

```
src/
  build_corpus.py        # parse upstream docs -> corpus/*.json
  zeiss_api_mcp.py       # MCP server
utils/
  mcp_poke.py            # debug / manual CLI client
corpus/                  # generated; gitignored
repos/                   # clones of the two ZEISS repos; gitignored
```
