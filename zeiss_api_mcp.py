#!/usr/bin/env python3
"""
zeiss_api_mcp.py — MCP server exposing the ZEISS INSPECT Python API corpus.

Run `build_corpus.py` first to generate the corpus/ directory, then point this
server at it:

    python zeiss_api_mcp.py --corpus ./corpus

Transports:
    --stdio         (default) - for local clients
    --http HOST:PORT          - for remote / systemd deployment
                                (e.g. --http 127.0.0.1:8765)

Tool surface:
    lookup_function(name)  signature, params, examples, howtos for a function
    lookup_class(name)     class description, method list, cross-refs
    lookup_module(name)    module description + function/class listing
    get_example(name)      full example doc + all scripts + API calls made
    get_howto(slug)        full how-to guide text (accepts dots or slashes)
    search(query, kind?)   BM25 search over names + descriptions + docs
                           (handles CamelCase: "scripted curve check" hits
                           ScriptedCurveCheck)
    search_by_tag(tag)     examples by tag (tags are derived from category +
                           name tokens if the source has none)
    list_modules()         all module names with function/class counts

Requires: pip install "mcp[cli]" rank_bm25
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
except ImportError:
    sys.exit("error: install with `pip install 'mcp[cli]'`")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("error: install with `pip install rank_bm25`")


# =============================================================================
# Tokenization — the single biggest fix in this file
# =============================================================================

# Split CamelCase: "ScriptedCurveCheck" -> "Scripted Curve Check".
# Two lookarounds handle both lower->upper and acronym boundaries
# ("XMLParser" -> "XML Parser").
_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
_WORD_RE = re.compile(r'[a-zA-Z0-9]+')


def _tokenize(text: str) -> list[str]:
    """Split on non-alphanumerics AND CamelCase boundaries, lowercased.

    "ScriptedCurveCheck"        -> ["scripted", "curve", "check"]
    "gom.api.scripted_checks"   -> ["gom", "api", "scripted", "checks"]
    "get_cs_transformation_4x4" -> ["get", "cs", "transformation", "4x4"]
    """
    if not text:
        return []
    pieces = _CAMEL_RE.sub(' ', text)
    return [t.lower() for t in _WORD_RE.findall(pieces)]


# =============================================================================
# Corpus loading
# =============================================================================

@dataclass
class Corpus:
    functions: dict[str, dict]
    classes: dict[str, dict]
    modules: dict[str, dict]
    examples: dict[str, dict]
    howtos: dict[str, dict]
    # derived:
    # normalized howto slug ("foo.bar" / "foo/bar" both -> "foo.bar") -> real slug
    howto_slug_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, corpus_dir: Path) -> "Corpus":
        def _j(name: str) -> dict:
            p = corpus_dir / name
            if not p.exists():
                print(f"warning: {p} missing; treating as empty", file=sys.stderr)
                return {}
            return json.loads(p.read_text(encoding="utf-8"))

        C = cls(
            functions=_j("api_functions.json"),
            classes=_j("api_classes.json"),
            modules=_j("modules.json"),
            examples=_j("examples.json"),
            howtos=_j("howtos.json"),
        )

        # --- Fix #2: populate tags on examples ---
        # Upstream has them empty, so search_by_tag is useless without this.
        for ex in C.examples.values():
            ex["tags"] = _derive_tags(ex)

        # --- Fix #3: build normalized slug map for howtos ---
        # Example docs link to "howtos/scripted_elements/scripted_checks.html"
        # but the index key is "scripted_elements.scripted_checks". Accept both.
        for slug in C.howtos:
            C.howto_slug_map[_normalize_slug(slug)] = slug

        # --- Sanity: flag fqns referenced by examples but missing from the
        # function index. This is an upstream crawler problem we can't fix
        # here, but we should know about it.
        _report_missing_api_calls(C)

        return C


def _derive_tags(ex: dict) -> list[str]:
    """Populate example tags from category + name tokens when upstream is empty.

    Keeps whatever tags the source already has, then adds:
      - the raw category ("scripted_checks")
      - a hyphenated singular-ish form ("scripted-check")
      - every CamelCase/underscore token from the name ("scripted", "curve",
        "check" for ScriptedCurveCheck).
    """
    tags = set(ex.get("tags") or [])
    cat = ex.get("category", "") or ""
    if cat:
        tags.add(cat)
        hyph = cat.replace("_", "-")
        tags.add(hyph)
        # crude singular: trailing 's' only (enough for scripted_checks ->
        # scripted-check; not doing full inflection).
        if hyph.endswith("s") and len(hyph) > 2:
            tags.add(hyph[:-1])
    tags.update(_tokenize(ex.get("name", "")))
    tags.discard("")
    return sorted(tags)


def _normalize_slug(s: str) -> str:
    """Treat '/' and '\\' as slug separators equivalent to '.'."""
    return s.replace("/", ".").replace("\\", ".").strip(".").lower()


def _report_missing_api_calls(C: Corpus) -> None:
    """Log fqns that examples/howtos reference but the function index lacks.

    This is diagnostic only — it flags crawler bugs in build_corpus.py,
    e.g. gom.api.scripted_checks_util.get_cs_transformation_4x4 being used
    in ScriptedCurveCheck but not indexed.
    """
    all_refs: set[str] = set()
    for ex in C.examples.values():
        all_refs.update(ex.get("api_calls", []) or [])
    # Heuristic: only care about gom.api.* calls (gom.Vec3d, gom.script.* are
    # not in the API reference). Allow prefix match because api_calls may
    # include attribute chains like "gom.api.foo.bar.baz" where "bar" is the
    # function.
    missing = set()
    fqn_set = set(C.functions)
    for ref in all_refs:
        if not ref.startswith("gom.api."):
            continue
        if ref in fqn_set:
            continue
        # try walking down prefixes — "gom.api.foo.bar.baz" might resolve to
        # the function "gom.api.foo.bar" with an attribute access after.
        parts = ref.split(".")
        resolved = False
        for i in range(len(parts), 2, -1):
            if ".".join(parts[:i]) in fqn_set:
                resolved = True
                break
        if not resolved:
            missing.add(ref)
    if missing:
        print(f"note: {len(missing)} gom.api.* references in examples not "
              f"found in function index (likely build_corpus.py crawler gap):",
              file=sys.stderr)
        for m in sorted(missing)[:10]:
            print(f"  missing: {m}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)


# =============================================================================
# Resolution: name -> fqn (handles partial / unqualified queries)
# =============================================================================

def _resolve_function(C: Corpus, name: str) -> list[str]:
    """Return list of fqns matching `name`. Exact fqn first, then suffix match."""
    if name in C.functions:
        return [name]
    matches = [fqn for fqn in C.functions if fqn == name or fqn.endswith("." + name)]
    return sorted(matches)


def _resolve_class(C: Corpus, name: str) -> list[str]:
    if name in C.classes:
        return [name]
    matches = [fqn for fqn in C.classes if fqn == name or fqn.endswith("." + name)]
    return sorted(matches)


def _resolve_module(C: Corpus, name: str) -> list[str]:
    if name in C.modules:
        return [name]
    matches = [fqn for fqn in C.modules if fqn == name or fqn.endswith("." + name)]
    return sorted(matches)


# =============================================================================
# BM25 search index — built once at startup
# =============================================================================

class SearchIndex:
    """Per-kind BM25 index over tokenized node content.

    The trick that fixes Qwen's multi-word misses: we build a bag of tokens
    per node from (name, signature, description, extended_description, and —
    for examples — the full documentation markdown). CamelCase names are
    decomposed before indexing, so 'scripted curve check' matches
    ScriptedCurveCheck at query time.
    """

    def __init__(self, C: Corpus):
        self.kinds: dict[str, tuple[list[str], BM25Okapi]] = {}

        self._build("function", C.functions, lambda k, v: (
            _tokenize(k)
            + _tokenize(v.get("description", ""))
            + _tokenize(v.get("extended_description", ""))
            + _tokenize(v.get("signature", ""))
        ))

        self._build("class", C.classes, lambda k, v: (
            _tokenize(k)
            + _tokenize(v.get("description", ""))
            + _tokenize(v.get("extended_description", ""))
        ))

        self._build("module", C.modules, lambda k, v: (
            _tokenize(k)
            + _tokenize(v.get("description", ""))
        ))

        self._build("example", C.examples, lambda k, v: (
            # Repeat name tokens to boost their weight — rank_bm25 has no
            # native field weighting, so duplicating is the cheap workaround.
            _tokenize(k) * 3
            + _tokenize(v.get("category", "")) * 2
            + _tokenize(v.get("description", ""))
            + _tokenize(v.get("documentation", ""))
            + [t.lower() for t in (v.get("tags") or [])]
            + _tokenize(" ".join(v.get("api_calls") or []))
        ))

        self._build("howto", C.howtos, lambda k, v: (
            _tokenize(k) * 3
            + _tokenize(v.get("title", "")) * 2
            + _tokenize(v.get("content", ""))
        ))

    def _build(
        self,
        kind: str,
        items: dict[str, dict],
        doc_fn: Callable[[str, dict], list[str]],
    ) -> None:
        keys: list[str] = []
        docs: list[list[str]] = []
        for k, v in items.items():
            toks = doc_fn(k, v)
            if toks:
                keys.append(k)
                docs.append(toks)
        if docs:
            self.kinds[kind] = (keys, BM25Okapi(docs))

    def query(self, kind: str, q: str, limit: int) -> list[str]:
        if kind not in self.kinds:
            return []
        keys, bm25 = self.kinds[kind]
        toks = _tokenize(q)
        if not toks:
            return []
        scores = bm25.get_scores(toks)
        ranked = sorted(
            ((s, k) for s, k in zip(scores, keys) if s > 0),
            key=lambda t: t[0],
            reverse=True,
        )[:limit]
        return [k for _, k in ranked]


# =============================================================================
# Response shapers (trim verbose fields for typical LLM consumption)
# =============================================================================

def _function_view(C: Corpus, fqn: str, verbose: bool = False) -> dict:
    """Compact function dict with embedded example snippets."""
    fn = C.functions[fqn]
    out = {
        "fqn": fn["fqn"],
        "module": fn["module"],
        "kind": fn["kind"],
        "signature": fn["signature"],
        "description": fn["description"],
        "params": fn["params"],
        "returns": fn["returns"],
        "return_type": fn["return_type"],
        "api_version": fn["api_version"],
    }
    if fn["class_fqn"]:
        out["class"] = fn["class_fqn"]
    if fn["extended_description"]:
        out["extended_description"] = fn["extended_description"]
    if fn["used_by_examples"]:
        out["used_by_examples"] = fn["used_by_examples"]
    if fn["mentioned_in_howtos"]:
        out["mentioned_in_howtos"] = fn["mentioned_in_howtos"]
    if verbose and fn.get("source_file"):
        out["source_file"] = fn["source_file"]
    return out


def _class_view(C: Corpus, fqn: str) -> dict:
    cls = C.classes[fqn]
    out = {
        "fqn": cls["fqn"],
        "module": cls["module"],
        "name": cls["name"],
        "description": cls["description"],
        "extended_description": cls["extended_description"],
        "methods": [
            {"fqn": m, "signature": C.functions[m]["signature"],
             "description": C.functions[m]["description"]}
            for m in cls["methods"] if m in C.functions
        ],
    }
    if cls.get("used_by_examples"):
        out["used_by_examples"] = cls["used_by_examples"]
    if cls.get("mentioned_in_howtos"):
        out["mentioned_in_howtos"] = cls["mentioned_in_howtos"]
    return out


def _module_view(C: Corpus, name: str) -> dict:
    mod = C.modules[name]
    out = {
        "name": mod["name"],
        "description": mod["description"],
        "functions": [
            {"fqn": f, "signature": C.functions[f]["signature"],
             "description": C.functions[f]["description"]}
            for f in mod["functions"] if f in C.functions
        ],
        "classes": [
            {"fqn": c, "name": C.classes[c]["name"],
             "description": C.classes[c]["description"],
             "method_count": len(C.classes[c]["methods"])}
            for c in mod["classes"] if c in C.classes
        ],
    }
    if mod.get("mentioned_in_howtos"):
        out["mentioned_in_howtos"] = mod["mentioned_in_howtos"]
    return out


def _example_view(C: Corpus, name: str, full_code: bool = True) -> dict:
    ex = C.examples[name]
    out = {
        "name": ex["name"],
        "category": ex["category"],
        "description": ex["description"],
        "tags": ex["tags"],
        "version_min": ex["version_min"],
        "path": ex["path"],
        "example_projects": ex["example_projects"],
        "api_calls": ex["api_calls"],
        "documentation": ex["documentation"],
    }
    if full_code:
        out["scripts"] = ex["scripts"]
    else:
        out["scripts"] = {k: f"[{len(v)} chars; request full_code=True to see]"
                          for k, v in ex["scripts"].items()}
    if ex.get("mentioned_in_howtos"):
        out["mentioned_in_howtos"] = ex["mentioned_in_howtos"]
    return out


def _howto_view(C: Corpus, slug: str) -> dict:
    ht = C.howtos[slug]
    return {
        "slug": ht["slug"],
        "title": ht["title"],
        "content": ht["content"],
        "api_mentions": ht["api_mentions"],
        "example_mentions": ht["example_mentions"],
        "linked_howtos": ht["linked_howtos"],
    }


# =============================================================================
# MCP server
# =============================================================================

def build_server(C: Corpus, transport_security=None) -> FastMCP:
    if transport_security is not None:
        mcp = FastMCP("zeiss-inspect-api", transport_security=transport_security)
    else:
        mcp = FastMCP("zeiss-inspect-api")

    # Build the BM25 indices once, closed over by the tool handlers.
    INDEX = SearchIndex(C)
    print(f"search index built for kinds: {sorted(INDEX.kinds)}", file=sys.stderr)

    @mcp.tool()
    def lookup_function(name: str) -> dict[str, Any]:
        """Look up a ZEISS INSPECT Python API function by name.

        Accepts full fqn (gom.api.settings.get) or partial (settings.get, get).
        Returns signature, parameters, return info, and cross-references to
        examples and how-to guides that use it.

        On ambiguous partial matches, returns {"candidates": [...]}.
        """
        matches = _resolve_function(C, name)
        if not matches:
            return {"error": f"no function matching {name!r}"}
        if len(matches) > 1:
            return {"candidates": [
                {"fqn": m, "signature": C.functions[m]["signature"],
                 "description": C.functions[m]["description"]}
                for m in matches
            ]}
        return _function_view(C, matches[0])

    @mcp.tool()
    def lookup_class(name: str) -> dict[str, Any]:
        """Look up a ZEISS INSPECT Python API class by name.

        Returns the class description, its methods (with signatures), and
        cross-references to examples and how-to guides. Accepts full fqn or
        short name (AddOn, Resource).
        """
        matches = _resolve_class(C, name)
        if not matches:
            return {"error": f"no class matching {name!r}"}
        if len(matches) > 1:
            return {"candidates": [
                {"fqn": m, "description": C.classes[m]["description"]}
                for m in matches
            ]}
        return _class_view(C, matches[0])

    @mcp.tool()
    def lookup_module(name: str) -> dict[str, Any]:
        """Look up a ZEISS INSPECT Python API module by name.

        Returns module description, list of module-level functions (name +
        signature + brief), and classes it contains. Accepts full fqn
        (gom.api.imaging) or short name (imaging).
        """
        matches = _resolve_module(C, name)
        if not matches:
            return {"error": f"no module matching {name!r}"}
        if len(matches) > 1:
            return {"candidates": matches}
        return _module_view(C, matches[0])

    @mcp.tool()
    def get_example(name: str, full_code: bool = True) -> dict[str, Any]:
        """Retrieve a full ZEISS INSPECT App example.

        Returns the documentation markdown, all Python scripts, tags, required
        version, and the list of gom.api.* calls the example makes. Set
        full_code=False to get script sizes without content.
        """
        if name in C.examples:
            return _example_view(C, name, full_code)
        nl = name.lower()
        matches = [n for n in C.examples if nl in n.lower()]
        if not matches:
            return {"error": f"no example matching {name!r}"}
        if len(matches) == 1:
            return _example_view(C, matches[0], full_code)
        return {"candidates": [
            {"name": m, "category": C.examples[m]["category"],
             "description": C.examples[m]["description"]}
            for m in sorted(matches)
        ]}

    @mcp.tool()
    def get_howto(slug: str) -> dict[str, Any]:
        """Retrieve the full content of a how-to guide.

        Slug format is the dotted path under doc/howtos/, e.g.
        "user_defined_dialogs.dialog_widgets". Slashes are also accepted:
        "scripted_elements/scripted_checks" resolves the same way. Partial
        matches are accepted.
        """
        # Exact: try as-given, then normalized (slash -> dot).
        if slug in C.howtos:
            return _howto_view(C, slug)
        normalized = _normalize_slug(slug)
        if normalized in C.howto_slug_map:
            return _howto_view(C, C.howto_slug_map[normalized])

        # Partial: substring match against normalized slugs.
        nl = normalized
        matches = [real for norm, real in C.howto_slug_map.items() if nl in norm]
        if not matches:
            return {"error": f"no howto matching {slug!r}"}
        if len(matches) == 1:
            return _howto_view(C, matches[0])
        return {"candidates": [
            {"slug": m, "title": C.howtos[m]["title"]}
            for m in sorted(matches)
        ]}

    @mcp.tool()
    def search(query: str, kind: str = "all", limit: int = 25) -> dict[str, Any]:
        """BM25 search over names, descriptions, and documentation bodies.

        Handles CamelCase: 'scripted curve check' matches ScriptedCurveCheck.
        Also searches the full example documentation and function
        extended_description, not just short summaries.

        kind: "all" | "function" | "class" | "module" | "example" | "howto"
        """
        out: dict[str, list] = {}
        wanted = (["function", "class", "module", "example", "howto"]
                  if kind == "all" else [kind])

        for k in wanted:
            hits = INDEX.query(k, query, limit)
            if k == "function":
                out["functions"] = [
                    {"fqn": f, "signature": C.functions[f]["signature"],
                     "description": C.functions[f]["description"]}
                    for f in hits
                ]
            elif k == "class":
                out["classes"] = [
                    {"fqn": f, "description": C.classes[f]["description"],
                     "method_count": len(C.classes[f].get("methods", []))}
                    for f in hits
                ]
            elif k == "module":
                out["modules"] = [
                    {"name": n, "description": C.modules[n]["description"],
                     "fn_count": len(C.modules[n].get("functions", [])),
                     "cls_count": len(C.modules[n].get("classes", []))}
                    for n in hits
                ]
            elif k == "example":
                out["examples"] = [
                    {"name": n, "category": C.examples[n]["category"],
                     "description": C.examples[n]["description"],
                     "tags": C.examples[n]["tags"]}
                    for n in hits
                ]
            elif k == "howto":
                out["howtos"] = [
                    {"slug": s, "title": C.howtos[s]["title"]}
                    for s in hits
                ]
        return out

    @mcp.tool()
    def search_by_tag(tag: str) -> dict[str, Any]:
        """List examples carrying a given tag.

        Tags include the raw category ("scripted_checks"), a hyphenated
        singular-ish form ("scripted-check"), and every CamelCase token of the
        example name ("scripted", "curve", "check" for ScriptedCurveCheck).
        Comparison is case-insensitive.
        """
        tl = tag.lower()
        hits = [
            {"name": ex["name"], "category": ex["category"],
             "description": ex["description"], "tags": ex["tags"]}
            for ex in C.examples.values()
            if any(t.lower() == tl for t in ex.get("tags", []))
        ]
        return {"tag": tag, "count": len(hits), "examples": hits}

    @mcp.tool()
    def list_modules() -> dict[str, Any]:
        """List all documented modules with function/class counts."""
        return {
            "modules": [
                {"name": name,
                 "description": mod["description"][:200],
                 "fn_count": len(mod.get("functions", [])),
                 "cls_count": len(mod.get("classes", []))}
                for name, mod in sorted(C.modules.items())
            ]
        }

    return mcp


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("corpus"),
                    help="directory containing *.json from build_corpus.py")
    ap.add_argument("--http", metavar="HOST:PORT",
                    help="serve over HTTP instead of stdio (e.g. 127.0.0.1:8765)")
    ap.add_argument("--allow-origin", action="append", default=[], metavar="ORIGIN",
                    help="add a trusted Origin (e.g. http://192.168.1.22:8082); "
                         "repeatable. Needed when a webui/CORS-proxy fronts the server.")
    ap.add_argument("--allow-host", action="append", default=[], metavar="HOST",
                    help="add a trusted Host header value (e.g. 192.168.1.22:8765); "
                         "repeatable.")
    ap.add_argument("--no-security", action="store_true",
                    help="disable DNS-rebinding protection entirely "
                         "(only do this on a trusted network).")
    args = ap.parse_args()

    if not args.corpus.is_dir():
        sys.exit(f"error: corpus directory {args.corpus} not found")

    C = Corpus.load(args.corpus)
    print(f"loaded: {len(C.functions)} functions, {len(C.classes)} classes, "
          f"{len(C.modules)} modules, {len(C.examples)} examples, "
          f"{len(C.howtos)} howtos", file=sys.stderr)

    security = None
    if args.http:
        if args.no_security:
            security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
            print("WARNING: DNS-rebinding protection disabled", file=sys.stderr)
        elif args.allow_origin or args.allow_host:
            security = TransportSecuritySettings(
                allowed_origins=args.allow_origin,
                allowed_hosts=args.allow_host,
            )
            if args.allow_origin:
                print(f"trusted origins: {args.allow_origin}", file=sys.stderr)
            if args.allow_host:
                print(f"trusted hosts: {args.allow_host}", file=sys.stderr)

    mcp = build_server(C, transport_security=security)

    if args.http:
        host, _, port = args.http.partition(":")
        mcp.settings.host = host or "127.0.0.1"
        mcp.settings.port = int(port) if port else 8765
        print(f"serving streamable-http on {mcp.settings.host}:{mcp.settings.port}",
              file=sys.stderr)
        mcp.run(transport="streamable-http")
    else:
        print("serving stdio", file=sys.stderr)
        mcp.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
