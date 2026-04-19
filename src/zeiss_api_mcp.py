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
    get_corpus_meta()      version + build provenance for the loaded corpus
    lookup_function(name)  signature, params, examples, howtos for a function
    get_function_examples(name, limit?)
                           full examples (with scripts) for a function, in one
                           call (collapses lookup_function -> get_example chain)
    lookup_class(name)     class description, method list, cross-refs
    lookup_module(name)    module description + function/class listing
    dump_module(name)      every function + class in a module, with full
                           descriptions (use instead of paginating search)
    list_all_symbols(...)  flat list of every documented symbol, optional
                           prefix/kind filter — for "what exists" exploration
    get_example(name)      full example doc + all scripts + API calls made
    get_howto(slug)        full how-to guide text (accepts dots or slashes)
    search(query, kind?, mode?)
                           hybrid BM25 + dense semantic search, fused with
                           reciprocal rank fusion. mode="hybrid"|"bm25"|
                           "semantic" (default hybrid; falls back to bm25
                           if sentence-transformers isn't installed).
    search_by_tag(tag)     examples by tag (tags are derived from category +
                           name tokens if the source has none)
    list_example_categories()
                           every category with its example names + one-liners
    list_modules()         all module names with function/class counts

Requires:
    pip install "mcp[cli]" rank_bm25
Optional (enables semantic + hybrid search):
    pip install sentence-transformers numpy
"""

from __future__ import annotations

import argparse
import hashlib
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
    # Optional side index of gom.* tokens referenced by example/howto source
    # but not present in the documented API (see build_corpus.build_referenced_symbols).
    # Absent on corpora built before that index existed — always guard with .get().
    referenced_symbols: dict[str, dict] = field(default_factory=dict)

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

        # Optional: load the referenced-symbols side index if the corpus
        # has one. Missing file is not an error — older corpora predate this.
        ref_path = corpus_dir / "referenced_symbols.json"
        if ref_path.exists():
            ref = json.loads(ref_path.read_text(encoding="utf-8"))
            C.referenced_symbols = ref.get("symbols", {}) or {}

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


def _referenced_but_undocumented(C: Corpus, name: str) -> dict[str, Any] | None:
    """Return a structured "not_documented" response for tokens that show up in
    example/howto source but aren't part of the documented API — or None.

    Prevents the common failure mode where the model lookups a symbol
    discovered via `search` that lives in the examples repo (e.g.
    `gom.script.sys.close_project`) and, on the empty miss, burns its
    tool budget retrying.
    """
    entry = C.referenced_symbols.get(name)
    if entry is None or entry.get("status") != "mentioned_only":
        return None
    return {
        "status": "not_documented",
        "name": name,
        "note": ("This symbol appears in example/howto source but is not "
                 "part of the documented API. Do not attempt further lookups."),
        "mentioned_in": entry.get("mentioned_in", []),
    }


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
# Semantic embedding index — optional, gracefully absent
# =============================================================================
#
# BM25 fails exactly when the user doesn't know the vocabulary. Dense
# embeddings catch those conceptual queries ("how do I move the camera in
# screen space?" -> finds viewport / projection helpers even though those
# words aren't in the docstring). We fuse both rankings with RRF so neither
# source dominates.
#
# Storage: one .npz per kind in <cache>/embeddings_<model>_<kind>.npz holding
# normalized float32 vectors + the keys list + a content signature. We
# rebuild a kind only if its signature changes, so adding one example doesn't
# re-encode the entire corpus.

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _truncate(s: str | None, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n]


def _embed_text_function(k: str, v: dict) -> str:
    return " | ".join(p for p in (
        k,
        v.get("signature", "") or "",
        v.get("description", "") or "",
        _truncate(v.get("extended_description", ""), 1500),
    ) if p)


def _embed_text_class(k: str, v: dict) -> str:
    return " | ".join(p for p in (
        k,
        v.get("description", "") or "",
        _truncate(v.get("extended_description", ""), 1500),
    ) if p)


def _embed_text_module(k: str, v: dict) -> str:
    return " | ".join(p for p in (
        k,
        _truncate(v.get("description", ""), 2000),
    ) if p)


def _embed_text_example(k: str, v: dict) -> str:
    return " | ".join(p for p in (
        v.get("name", k) or k,
        v.get("category", "") or "",
        v.get("description", "") or "",
        _truncate(v.get("documentation", ""), 1500),
    ) if p)


def _embed_text_howto(k: str, v: dict) -> str:
    return " | ".join(p for p in (
        v.get("title", k) or k,
        _truncate(v.get("content", ""), 2500),
    ) if p)


_EMBED_BUILDERS: dict[str, Callable[[str, dict], str]] = {
    "function": _embed_text_function,
    "class": _embed_text_class,
    "module": _embed_text_module,
    "example": _embed_text_example,
    "howto": _embed_text_howto,
}


def _safe_model_tag(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def _signature(keys: list[str], texts: list[str]) -> str:
    h = hashlib.sha256()
    for k, t in zip(keys, texts):
        h.update(k.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x01")
    return h.hexdigest()


class SemanticIndex:
    """Per-kind dense embedding index with on-disk caching.

    Loads sentence-transformers lazily so the rest of the server keeps
    working when it isn't installed. `available` is False if anything
    goes wrong; callers should check it before using `query`.
    """

    def __init__(
        self,
        C: "Corpus",
        model_name: str = DEFAULT_EMBED_MODEL,
        cache_dir: Path | None = None,
        rebuild: bool = False,
        device: str | None = None,
    ) -> None:
        self.available = False
        self.model_name = model_name
        self.kinds: dict[str, tuple[list[str], "Any"]] = {}
        self._model = None

        try:
            import numpy as np  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as e:
            print(f"note: semantic search disabled ({e}); install with "
                  f"`pip install sentence-transformers numpy` to enable.",
                  file=sys.stderr)
            return

        self._np = __import__("numpy")
        self._SentenceTransformer = SentenceTransformer

        # device='auto' (the default) lets sentence-transformers pick CUDA if
        # torch.cuda.is_available(). That heuristic is wrong on machines whose
        # GPU is too old for the installed PyTorch wheel (e.g. a Tesla M40 at
        # sm_52 against a wheel built for sm_70+); torch reports the device
        # available but the first kernel launch fails. We resolve "auto" to a
        # safer choice up front and still catch any encode-time failure with
        # a CPU retry below.
        resolved = self._resolve_device(device)
        try:
            self._model = SentenceTransformer(model_name, device=resolved)
        except Exception as e:
            print(f"warning: failed to load embedding model {model_name!r} on "
                  f"{resolved}: {e}; continuing with BM25 only.", file=sys.stderr)
            return
        self._device = resolved

        sources = {
            "function": C.functions,
            "class": C.classes,
            "module": C.modules,
            "example": C.examples,
            "howto": C.howtos,
        }
        cache_dir = cache_dir or Path(".")
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = _safe_model_tag(model_name)

        for kind, items in sources.items():
            keys = list(items)
            if not keys:
                continue
            texts = [_EMBED_BUILDERS[kind](k, items[k]) for k in keys]
            sig = _signature(keys, texts)
            cache_file = cache_dir / f"embeddings_{tag}_{kind}.npz"

            vecs = None
            if cache_file.exists() and not rebuild:
                try:
                    data = self._np.load(cache_file, allow_pickle=False)
                    if (str(data["signature"]) == sig
                            and list(data["keys"]) == keys):
                        vecs = data["vecs"]
                except Exception as e:
                    print(f"note: cache {cache_file.name} unusable ({e}); "
                          f"recomputing.", file=sys.stderr)

            if vecs is None:
                print(f"embedding {len(keys)} {kind}s with {model_name} "
                      f"on {self._device}...", file=sys.stderr)
                vecs = self._encode_with_fallback(texts)
                if vecs is None:
                    return  # already logged; fall back to BM25-only
                try:
                    self._np.savez(
                        cache_file,
                        keys=self._np.array(keys),
                        vecs=vecs,
                        signature=self._np.array(sig),
                    )
                except Exception as e:
                    print(f"warning: failed to save {cache_file.name}: {e}",
                          file=sys.stderr)

            self.kinds[kind] = (keys, vecs)

        self.available = bool(self.kinds)
        if self.available:
            print(f"semantic index ready ({model_name} on {self._device}): "
                  f"{', '.join(f'{k}={len(v[0])}' for k, v in self.kinds.items())}",
                  file=sys.stderr)

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        """Pick a device for SentenceTransformer.

        Explicit values ('cpu', 'cuda', 'cuda:1', 'mps', ...) pass through.
        'auto' / None probes torch: prefer CUDA only if the device's compute
        capability is actually supported by the installed wheel; otherwise
        fall back to CPU and log why.
        """
        if device and device != "auto":
            return device
        try:
            import torch
        except Exception:
            return "cpu"
        if not torch.cuda.is_available():
            return "cpu"
        try:
            major, minor = torch.cuda.get_device_capability(0)
            supported = set(torch.cuda.get_arch_list() or [])
            wanted = f"sm_{major}{minor}"
            if supported and wanted not in supported:
                name = torch.cuda.get_device_name(0)
                print(f"note: {name} ({wanted}) is not in this PyTorch's "
                      f"supported archs ({sorted(supported)}); using CPU for "
                      f"embeddings. Pass --embedding-device cuda to override.",
                      file=sys.stderr)
                return "cpu"
        except Exception:
            return "cpu"
        return "cuda"

    def _encode_with_fallback(self, texts: list[str]):
        """Encode `texts`, retrying on CPU if a CUDA error trips the GPU.

        Some installs report `cuda.is_available()` true but explode at the
        first kernel launch (sm mismatch, bad driver, OOM during init). We
        catch the first failure, rebuild the model on CPU, and try again so
        the server keeps working without the user having to restart.
        """
        try:
            return self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            ).astype(self._np.float32)
        except Exception as e:
            if self._device == "cpu":
                print(f"warning: encode failed on CPU ({e}); "
                      f"semantic search disabled.", file=sys.stderr)
                return None
            print(f"warning: encode failed on {self._device} ({e}); "
                  f"retrying on CPU. Pass --embedding-device cpu to skip "
                  f"this probe next time.", file=sys.stderr)
            try:
                self._model = self._SentenceTransformer(
                    self.model_name, device="cpu"
                )
                self._device = "cpu"
                return self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=32,
                ).astype(self._np.float32)
            except Exception as e2:
                print(f"warning: CPU retry also failed ({e2}); "
                      f"semantic search disabled.", file=sys.stderr)
                return None

    def query(self, kind: str, q: str, limit: int) -> list[str]:
        if not self.available or kind not in self.kinds or not q.strip():
            return []
        keys, vecs = self.kinds[kind]
        q_vec = self._model.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(self._np.float32)
        scores = vecs @ q_vec  # cosine because both sides are unit-normalized
        # argpartition for the top-k, then exact-sort just those.
        n = scores.shape[0]
        if limit >= n:
            order = self._np.argsort(-scores)
        else:
            top_idx = self._np.argpartition(-scores, limit)[:limit]
            order = top_idx[self._np.argsort(-scores[top_idx])]
        return [keys[int(i)] for i in order if scores[int(i)] > 0.0]


# =============================================================================
# Hybrid retrieval: RRF over BM25 + semantic
# =============================================================================
#
# Reciprocal rank fusion: score(d) = sum_r 1 / (k + rank_r(d)). Each ranker
# contributes independently; a doc that's first in BM25 and absent from
# semantic still gets a credible score, and vice versa. k=60 is the value
# from the original Cormack/Clarke/Buettcher paper and is robust enough
# that we don't bother tuning it.

def _rrf_fuse(rankings: list[list[str]], limit: int, k: int = 60
              ) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(scores.items(),
                                 key=lambda t: -t[1])[:limit]]


class HybridSearchIndex:
    """Wraps BM25 (precision on known names) and semantic (recall on
    conceptual queries) with three explicit modes. `hybrid` is the default
    when both are available; otherwise it transparently degrades to BM25.
    """

    def __init__(self, bm25: "SearchIndex", semantic: SemanticIndex) -> None:
        self.bm25 = bm25
        self.semantic = semantic

    @property
    def has_semantic(self) -> bool:
        return self.semantic.available

    def query(
        self,
        kind: str,
        q: str,
        limit: int,
        mode: str = "hybrid",
    ) -> list[str]:
        if mode == "bm25" or not self.has_semantic:
            return self.bm25.query(kind, q, limit)
        if mode == "semantic":
            return self.semantic.query(kind, q, limit)
        # hybrid: pull a wider candidate pool from both, fuse by RRF.
        pool = max(limit * 4, 50)
        bm = self.bm25.query(kind, q, pool)
        sem = self.semantic.query(kind, q, pool)
        if not bm:
            return sem[:limit]
        if not sem:
            return bm[:limit]
        return _rrf_fuse([bm, sem], limit)


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

def build_server(
    C: Corpus,
    corpus_dir: Path | None = None,
    transport_security=None,
    embed_model: str | None = DEFAULT_EMBED_MODEL,
    embed_cache_dir: Path | None = None,
    rebuild_embeddings: bool = False,
    embed_device: str | None = None,
) -> FastMCP:
    if transport_security is not None:
        mcp = FastMCP("zeiss-inspect-api", transport_security=transport_security)
    else:
        mcp = FastMCP("zeiss-inspect-api")

    # Build the BM25 indices once, closed over by the tool handlers.
    BM25 = SearchIndex(C)
    print(f"bm25 index built for kinds: {sorted(BM25.kinds)}", file=sys.stderr)

    if embed_model:
        SEM = SemanticIndex(
            C,
            model_name=embed_model,
            cache_dir=embed_cache_dir,
            rebuild=rebuild_embeddings,
            device=embed_device,
        )
    else:
        SEM = SemanticIndex.__new__(SemanticIndex)
        SEM.available = False
        SEM.kinds = {}
        SEM.model_name = ""
        print("semantic search disabled by --no-semantic", file=sys.stderr)
    INDEX = HybridSearchIndex(BM25, SEM)

    @mcp.tool()
    def get_corpus_meta() -> dict[str, Any]:
        """Return version and build provenance for the loaded corpus.

        Call this first to confirm which ZEISS INSPECT version you're grounded in.
        """
        if corpus_dir is None:
            return {"note": "corpus directory not provided to server"}
        p = corpus_dir / "corpus_meta.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"note": "corpus_meta.json not found; rebuild corpus to generate it"}

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
            undoc = _referenced_but_undocumented(C, name)
            if undoc is not None:
                return undoc
            return {"error": f"no function matching {name!r}"}
        if len(matches) > 1:
            return {"candidates": [
                {"fqn": m, "signature": C.functions[m]["signature"],
                 "description": C.functions[m]["description"]}
                for m in matches
            ]}
        return _function_view(C, matches[0])

    @mcp.tool()
    def get_function_examples(name: str, limit: int = 3) -> dict[str, Any]:
        """Return full examples (with scripts) for all examples that use a given function.

        Collapses the common lookup_function -> get_example chain into one call.
        Accepts full fqn or partial name.
        """
        matches = _resolve_function(C, name)
        if not matches:
            return {"error": f"no function matching {name!r}"}
        if len(matches) > 1:
            return {"candidates": [{"fqn": m} for m in matches]}
        fn = C.functions[matches[0]]
        names = (fn.get("used_by_examples") or [])[:limit]
        return {
            "fqn": matches[0],
            "example_count": len(fn.get("used_by_examples") or []),
            "examples": [_example_view(C, n) for n in names if n in C.examples],
        }

    @mcp.tool()
    def lookup_class(name: str) -> dict[str, Any]:
        """Look up a ZEISS INSPECT Python API class by name.

        Returns the class description, its methods (with signatures), and
        cross-references to examples and how-to guides. Accepts full fqn or
        short name (AddOn, Resource).
        """
        matches = _resolve_class(C, name)
        if not matches:
            undoc = _referenced_but_undocumented(C, name)
            if undoc is not None:
                return undoc
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
    def search(
        query: str,
        kind: str = "all",
        limit: int = 25,
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Hybrid (BM25 + dense semantic) search across the API corpus.

        Handles CamelCase: 'scripted curve check' matches ScriptedCurveCheck.
        Searches names, signatures, full example documentation, and function
        extended_description — not just short summaries.

        kind: "all" | "function" | "class" | "module" | "example" | "howto"
        mode: "hybrid" (default; reciprocal-rank-fuses BM25 + semantic) |
              "bm25" (lexical only; best for known FQNs / exact tokens) |
              "semantic" (dense only; best for paraphrase / concept queries).
              Falls back to bm25 if sentence-transformers isn't installed.
        """
        if mode not in ("hybrid", "bm25", "semantic"):
            return {"error": f"mode must be hybrid|bm25|semantic, got {mode!r}"}
        out: dict[str, Any] = {
            "mode": mode if INDEX.has_semantic or mode == "bm25" else "bm25",
            "semantic_available": INDEX.has_semantic,
        }
        wanted = (["function", "class", "module", "example", "howto"]
                  if kind == "all" else [kind])

        for k in wanted:
            hits = INDEX.query(k, query, limit, mode=mode)
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
    def list_example_categories() -> dict[str, Any]:
        """List all example categories with their examples.

        Use this for 'what scripted_checks examples exist?' style queries.
        """
        cats: dict[str, list[dict]] = {}
        for ex in C.examples.values():
            cat = ex.get("category") or "uncategorized"
            cats.setdefault(cat, []).append({
                "name": ex["name"],
                "description": ex["description"],
            })
        return {"categories": {k: sorted(v, key=lambda x: x["name"])
                               for k, v in sorted(cats.items())}}

    @mcp.tool()
    def dump_module(name: str, include_extended: bool = False) -> dict[str, Any]:
        """Dump every function and class in a module with full descriptions.

        Use this when you need to see everything a module exposes at once
        rather than paginating through search hits. Heavier than
        lookup_module: includes parameter lists, return info, and (optionally)
        extended descriptions for every member.

        Accepts full fqn (gom.api.imaging) or short name (imaging).
        """
        matches = _resolve_module(C, name)
        if not matches:
            return {"error": f"no module matching {name!r}"}
        if len(matches) > 1:
            return {"candidates": matches}

        fqn = matches[0]
        mod = C.modules[fqn]

        functions = []
        for f in mod.get("functions", []):
            if f not in C.functions:
                continue
            fn = C.functions[f]
            item = {
                "fqn": f,
                "signature": fn["signature"],
                "description": fn["description"],
                "params": fn["params"],
                "returns": fn["returns"],
                "return_type": fn["return_type"],
            }
            if include_extended and fn.get("extended_description"):
                item["extended_description"] = fn["extended_description"]
            functions.append(item)

        classes = []
        for c in mod.get("classes", []):
            if c not in C.classes:
                continue
            cls = C.classes[c]
            item = {
                "fqn": c,
                "name": cls["name"],
                "description": cls["description"],
                "methods": [
                    {"fqn": m, "signature": C.functions[m]["signature"],
                     "description": C.functions[m]["description"]}
                    for m in cls["methods"] if m in C.functions
                ],
            }
            if include_extended and cls.get("extended_description"):
                item["extended_description"] = cls["extended_description"]
            classes.append(item)

        return {
            "name": fqn,
            "description": mod["description"],
            "function_count": len(functions),
            "class_count": len(classes),
            "functions": functions,
            "classes": classes,
        }

    @mcp.tool()
    def list_all_symbols(
        prefix: str = "",
        kind: str = "all",
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Flat enumeration of every documented symbol (name + one-liner).

        Use this for "what exists?" exploration when you don't have a query
        good enough for search. Optional `prefix` is a case-insensitive
        substring filter applied to the name/fqn — pass "imaging" to see
        every symbol whose fqn or name contains "imaging".

        kind: "all" | "function" | "class" | "module" | "example" | "howto"
        limit: per-kind cap (default 1000; bump for large dumps).
        """
        pl = (prefix or "").lower()
        kinds = (["function", "class", "module", "example", "howto"]
                 if kind == "all" else [kind])
        out: dict[str, Any] = {"prefix": prefix, "kind": kind}

        if "function" in kinds:
            hits = sorted(f for f in C.functions if pl in f.lower())[:limit]
            out["functions"] = [
                {"fqn": f, "description": C.functions[f]["description"]}
                for f in hits
            ]
        if "class" in kinds:
            hits = sorted(c for c in C.classes if pl in c.lower())[:limit]
            out["classes"] = [
                {"fqn": c, "description": C.classes[c]["description"]}
                for c in hits
            ]
        if "module" in kinds:
            hits = sorted(m for m in C.modules if pl in m.lower())[:limit]
            out["modules"] = [
                {"name": m, "description": C.modules[m]["description"][:200]}
                for m in hits
            ]
        if "example" in kinds:
            hits = sorted(e for e in C.examples if pl in e.lower())[:limit]
            out["examples"] = [
                {"name": e, "category": C.examples[e]["category"],
                 "description": C.examples[e]["description"]}
                for e in hits
            ]
        if "howto" in kinds:
            hits = sorted(s for s in C.howtos if pl in s.lower())[:limit]
            out["howtos"] = [
                {"slug": s, "title": C.howtos[s]["title"]}
                for s in hits
            ]

        out["total"] = sum(
            len(v) for k, v in out.items() if isinstance(v, list)
        )
        return out

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
    ap.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL,
                    metavar="NAME",
                    help=f"sentence-transformers model for semantic search "
                         f"(default: {DEFAULT_EMBED_MODEL}). bge-small is ~33M "
                         f"params and runs comfortably on CPU; switch to "
                         f"nomic-ai/nomic-embed-text-v1.5 for higher quality "
                         f"if you have GPU headroom.")
    ap.add_argument("--embedding-cache-dir", type=Path, default=None,
                    metavar="DIR",
                    help="where to store cached embedding vectors "
                         "(default: <corpus-dir>/.embeddings).")
    ap.add_argument("--rebuild-embeddings", action="store_true",
                    help="ignore cached vectors and recompute everything.")
    ap.add_argument("--embedding-device", default="auto", metavar="DEV",
                    help="device for the embedding model: auto|cpu|cuda|cuda:N|"
                         "mps (default: auto). 'auto' probes the GPU's compute "
                         "capability against the installed PyTorch wheel and "
                         "falls back to CPU if they don't match (e.g. Tesla "
                         "M40 sm_52 against a wheel built for sm_70+).")
    ap.add_argument("--no-semantic", action="store_true",
                    help="disable dense semantic search; serve BM25 only.")
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

    cache_dir = args.embedding_cache_dir or (args.corpus / ".embeddings")
    mcp = build_server(
        C,
        corpus_dir=args.corpus,
        transport_security=security,
        embed_model=None if args.no_semantic else args.embedding_model,
        embed_cache_dir=cache_dir,
        rebuild_embeddings=args.rebuild_embeddings,
        embed_device=args.embedding_device,
    )

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
