#!/usr/bin/env python3
"""
build_corpus.py — Compile ZEISS INSPECT App API docs + examples into LLM-ready JSON.

Typical usage:
  python build_corpus.py --zeiss-version 2026
      Clones both ZEISS repos into ./repos/ (if not already there), checks out
      the year-matching release branch, and builds the corpus under ./corpus/.

Advanced:
  --workspace DIR      Where cloned repos live (default: ./repos)
  --api-repo DIR       Override api repo path (skip auto-clone)
  --ex-repo  DIR       Override examples repo path (skip auto-clone)
  --out      DIR       Output directory for the corpus JSON (default: ./corpus)
  --no-fetch           Don't `git fetch` before checkout (offline mode)

Branch strategy:
  The api repo uses year-named branches (2023, 2025, 2026, ...). `main` tracks
  the currently-shipping version and is usually one cycle behind the latest
  docs site. The examples repo may or may not use the same convention; we try
  the year branch first and fall back to `main` if it doesn't exist.

Outputs:
  api_functions.json   {fqn: ApiFunction}   functions, methods, static methods
  api_classes.json     {fqn: ApiClass}      class descriptions + method lists
  modules.json         {module: ModuleMeta} module description + class/function lists
  howtos.json          {slug: HowTo}        doc/howtos/**/*.md
  examples.json        {name: Example}      AppExamples/**/Documentation.md + code
  corpus_meta.json     {zeiss_version, built_at, repo commits}

Doc conventions handled (from actual python_api.md / resource_api.md inspection):
  - MyST {py:function} fence with brief+fields inside  -> direct parse
  - MyST {py:function} fence with field list only, brief/prose AFTER fence -> fallback
  - RST {eval-rst} + .. py:class / .. py:method (resource_api.md style)
  - H3 class headers WITHOUT a fence (e.g. MultiElementCreationScope) -> separate pass
  - Examples as '#### Example', '**Example**', '**Usage example**' etc
    -> captured as part of extended_description (not separately extracted)
  - CRLF line endings throughout
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# Repo preparation (clone + branch checkout)
# =============================================================================

API_REPO_URL = "https://github.com/ZEISS/zeiss-inspect-app-api.git"
EX_REPO_URL = "https://github.com/ZEISS/zeiss-inspect-app-examples.git"


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command for transparency. Captures output."""
    print(f"  $ {' '.join(cmd)}" + (f"   (in {cwd})" if cwd else ""), file=sys.stderr)
    return subprocess.run(
        cmd, cwd=cwd, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _remote_has_branch(url: str, branch: str) -> bool:
    """Check whether a given branch exists on a remote without cloning first."""
    r = _run(["git", "ls-remote", "--heads", url, branch], check=False)
    return bool(r.stdout.strip())


def _git_rev(repo: Path) -> str:
    """Current commit sha of a repo, short form. Empty string on failure."""
    try:
        r = _run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], check=False)
        return r.stdout.strip()
    except FileNotFoundError:
        return ""


def _ensure_repo(url: str, dest: Path, branch: str, fetch: bool = True) -> str:
    """Clone `url` to `dest` if missing, then check out `branch`.

    Returns the short commit sha of HEAD after checkout. Raises if the branch
    cannot be resolved (after optionally trying a `main` fallback for the
    examples repo, which is handled at the call site).
    """
    if not dest.exists():
        print(f"  cloning {url} -> {dest}", file=sys.stderr)
        _run(["git", "clone", "--branch", branch, url, str(dest)])
    else:
        if fetch:
            _run(["git", "-C", str(dest), "fetch", "origin", "--prune"])
        # Detect uncommitted changes and bail rather than silently clobbering.
        status = _run(["git", "-C", str(dest), "status", "--porcelain"], check=False)
        if status.stdout.strip():
            sys.exit(
                f"error: {dest} has uncommitted changes. Commit, stash, or remove "
                f"it before running.\n{status.stdout}"
            )
        _run(["git", "-C", str(dest), "checkout", branch])
        if fetch:
            # Fast-forward if possible; ignore failures (e.g. detached or diverged).
            _run(["git", "-C", str(dest), "pull", "--ff-only", "origin", branch], check=False)

    return _git_rev(dest)


def prepare_repos(
    workspace: Path,
    zeiss_version: str,
    fetch: bool = True,
    api_override: Path | None = None,
    ex_override: Path | None = None,
    ex_branch_override: str | None = None,
) -> tuple[Path, Path, dict]:
    """Resolve both repos to the right branch for `zeiss_version`.

    Returns (api_repo_path, ex_repo_path, metadata_dict).

    When `api_override` / `ex_override` are provided, those paths are used
    as-is with no clone/fetch (trusting the caller).

    Branch selection for the examples repo, in order:
      1. `ex_branch_override` if given (bypasses all detection).
      2. A branch literally named `<zeiss_version>` if it exists on origin.
      3. `main`. We warn when we fall back to main because the examples repo
         can lag the api repo — e.g. the 2026 api branch has custom-element
         howtos whose matching Python examples live on a feature branch like
         `20251211-feat-custom-elements` that hasn't been merged to main yet.
         In that case pass --ex-branch explicitly.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    meta = {
        "zeiss_version": zeiss_version,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # --- API repo ---
    if api_override is not None:
        if not api_override.is_dir():
            sys.exit(f"error: --api-repo {api_override} does not exist")
        api_path = api_override
        meta["api_repo"] = {"path": str(api_path), "branch": "(user-provided)",
                            "commit": _git_rev(api_path)}
    else:
        if not _remote_has_branch(API_REPO_URL, zeiss_version):
            sys.exit(
                f"error: zeiss-inspect-app-api has no '{zeiss_version}' branch on origin.\n"
                f"       check `git ls-remote --heads {API_REPO_URL}` for available years."
            )
        api_path = workspace / "zeiss-inspect-app-api"
        commit = _ensure_repo(API_REPO_URL, api_path, zeiss_version, fetch=fetch)
        meta["api_repo"] = {"path": str(api_path), "branch": zeiss_version, "commit": commit}

    # --- Examples repo ---
    if ex_override is not None:
        if not ex_override.is_dir():
            sys.exit(f"error: --ex-repo {ex_override} does not exist")
        ex_path = ex_override
        meta["ex_repo"] = {"path": str(ex_path), "branch": "(user-provided)",
                           "commit": _git_rev(ex_path)}
    else:
        # Resolve the branch to clone.
        if ex_branch_override is not None:
            if not _remote_has_branch(EX_REPO_URL, ex_branch_override):
                sys.exit(f"error: zeiss-inspect-app-examples has no branch "
                         f"{ex_branch_override!r} on origin.")
            ex_branch = ex_branch_override
            print(f"  using explicit examples branch: {ex_branch}", file=sys.stderr)
        elif _remote_has_branch(EX_REPO_URL, zeiss_version):
            ex_branch = zeiss_version
        else:
            ex_branch = "main"
            # Look for any feature branch that matches the current year — don't
            # pick it automatically, just tell the user it exists.
            r = _run(["git", "ls-remote", "--heads", EX_REPO_URL], check=False)
            feat_pat = re.compile(rf"^{zeiss_version[:4]}\d{{4}}-feat-")
            feat_branches = []
            for line in r.stdout.splitlines():
                if not line.strip():
                    continue
                _sha, ref = line.split(None, 1)
                branch = ref.replace("refs/heads/", "").strip()
                if feat_pat.match(branch):
                    feat_branches.append(branch)
            print(f"  note: zeiss-inspect-app-examples has no '{zeiss_version}' "
                  f"branch; using 'main'", file=sys.stderr)
            if feat_branches:
                print(f"  note: found {len(feat_branches)} '{zeiss_version[:4]}*-feat-*' "
                      f"feature branch(es) that may have newer examples:",
                      file=sys.stderr)
                for b in feat_branches:
                    print(f"        {b}", file=sys.stderr)
                print(f"        pass --ex-branch <name> to use one of these.",
                      file=sys.stderr)

        ex_path = workspace / "zeiss-inspect-app-examples"
        commit = _ensure_repo(EX_REPO_URL, ex_path, ex_branch, fetch=fetch)
        meta["ex_repo"] = {"path": str(ex_path), "branch": ex_branch, "commit": commit}

    return api_path, ex_path, meta


# =============================================================================
# Data model
# =============================================================================

@dataclass
class ApiFunction:
    fqn: str                                    # gom.api.addons.AddOn.exists
    module: str                                 # gom.api.addons
    name: str                                   # exists
    kind: str = "function"                      # function|method|staticmethod|classmethod
    class_name: str = ""                        # "AddOn" if a method (leaf name only)
    class_fqn: str = ""                         # "gom.api.addons.AddOn" if a method
    signature: str = ""
    params: list[dict[str, str]] = field(default_factory=list)
    returns: str = ""
    return_type: str = ""
    description: str = ""                       # brief
    extended_description: str = ""              # remaining prose (includes example blocks)
    api_version: str = ""
    source_file: str = ""
    source_anchor: str = ""
    used_by_examples: list[str] = field(default_factory=list)
    mentioned_in_howtos: list[str] = field(default_factory=list)


@dataclass
class ApiClass:
    fqn: str
    module: str
    name: str
    description: str = ""
    extended_description: str = ""
    methods: list[str] = field(default_factory=list)
    source_file: str = ""
    used_by_examples: list[str] = field(default_factory=list)
    mentioned_in_howtos: list[str] = field(default_factory=list)


@dataclass
class ModuleMeta:
    name: str
    description: str = ""
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    used_by_examples: list[str] = field(default_factory=list)
    mentioned_in_howtos: list[str] = field(default_factory=list)


@dataclass
class HowTo:
    slug: str
    title: str
    source_file: str = ""
    content: str = ""
    api_mentions: list[str] = field(default_factory=list)
    example_mentions: list[str] = field(default_factory=list)
    linked_howtos: list[str] = field(default_factory=list)


@dataclass
class Example:
    name: str
    category: str
    path: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    version_min: str = ""
    api_refs: list[str] = field(default_factory=list)
    howto_refs: list[str] = field(default_factory=list)
    api_calls: list[str] = field(default_factory=list)
    documentation: str = ""
    scripts: dict[str, str] = field(default_factory=dict)
    example_projects: list[str] = field(default_factory=list)
    mentioned_in_howtos: list[str] = field(default_factory=list)


# =============================================================================
# Text normalization
# =============================================================================

def _normalize(text: str) -> str:
    """Normalize CRLF -> LF. All downstream regexes assume LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


# =============================================================================
# MyST/Sphinx field list parsing
# =============================================================================

def _split_brief_and_fields(body: str) -> tuple[str, str]:
    """Brief = text before first `:field:` line. Fields = everything from there on."""
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^\s*:[A-Za-z]", line):
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i:])
    return body.strip(), ""


def _parse_fields(field_text: str) -> list[tuple[str, str]]:
    """Parse `:key [name]: value` lines, supporting multi-line continuations."""
    out: list[tuple[str, str]] = []
    cur_key: str | None = None
    cur_buf: list[str] = []
    for raw in field_text.split("\n"):
        line = raw.strip()
        m = re.match(r"^:\s*([^:]+?)\s*:\s*(.*)$", line)
        if m:
            if cur_key is not None:
                out.append((cur_key, "\n".join(cur_buf).strip()))
            cur_key = m.group(1).strip()
            cur_buf = [m.group(2)]
        elif cur_key is not None:
            cur_buf.append(raw)
    if cur_key is not None:
        out.append((cur_key, "\n".join(cur_buf).strip()))
    return out


def _organize_fields(fields: list[tuple[str, str]]) -> dict:
    out: dict = {"api_version": "", "params": [], "returns": "", "return_type": "",
                 "raises": [], "other": {}}
    params_by_name: dict[str, dict[str, str]] = {}
    for key, val in fields:
        k = key.lower().strip()
        if k == "api version":
            out["api_version"] = val
        elif k.startswith("param ") or k.startswith("parameter "):
            pname = key.split(None, 1)[1].strip()
            p = params_by_name.setdefault(pname, {"name": pname, "type": "", "desc": ""})
            p["desc"] = val
        elif k.startswith("type "):
            pname = key.split(None, 1)[1].strip()
            p = params_by_name.setdefault(pname, {"name": pname, "type": "", "desc": ""})
            p["type"] = val
        elif k in ("return", "returns"):
            out["returns"] = val
        elif k == "rtype":
            out["return_type"] = val
        elif k == "raises":
            out["raises"].append(val)
        else:
            out["other"][key] = val
    out["params"] = list(params_by_name.values())
    return out


def _promote_brief(description: str, extended: str) -> tuple[str, str]:
    """If description is empty but extended has content, promote its first paragraph."""
    if description or not extended:
        return description, extended
    parts = extended.split("\n\n", 1)
    first = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return first, rest


# =============================================================================
# Signature parsing
# =============================================================================

SIG_RE = re.compile(r"^(?P<fqn>[\w\.]+)\s*\((?P<params>.*)\)\s*(?::\s*(?P<ret>.+))?\s*$")


def _parse_sig_params(raw: str) -> list[dict[str, str]]:
    if not raw.strip():
        return []
    out: list[dict[str, str]] = []
    depth = 0
    buf = ""
    for ch in raw + ",":
        if ch in "([{<":
            depth += 1; buf += ch
        elif ch in ")]}>":
            depth -= 1; buf += ch
        elif ch == "," and depth == 0:
            p = buf.strip()
            buf = ""
            if not p:
                continue
            if "=" in p:
                p = p.split("=", 1)[0].strip()
            if ":" in p:
                n, t = p.split(":", 1)
                out.append({"name": n.strip(), "type": t.strip()})
            else:
                out.append({"name": p, "type": ""})
        else:
            buf += ch
    return out


def _merge_params(sig_params: list[dict], field_params: list[dict]) -> list[dict]:
    fb = {p["name"]: p for p in field_params}
    merged: list[dict] = []
    seen: set[str] = set()
    for sp in sig_params:
        fp = fb.get(sp["name"], {})
        merged.append({
            "name": sp["name"],
            "type": sp.get("type") or fp.get("type", ""),
            "desc": fp.get("desc", ""),
        })
        seen.add(sp["name"])
    for fp in field_params:
        if fp["name"] not in seen:
            merged.append({"name": fp["name"], "type": fp.get("type", ""),
                           "desc": fp.get("desc", "")})
    return merged


# Populated per-build from the H2/H3/H4 module-style headers in the doc pages.
# Longest-prefix match against this set is what lets `_module_of` resolve
# submodule names like `gom.api.extensions.actuals` instead of collapsing them
# into the nearest top-level module.
_KNOWN_MODULES: set[str] = set()


def _module_of(fqn: str) -> str:
    """Return the module part of a fully-qualified name.

    Prefers the longest prefix found in `_KNOWN_MODULES` so submodules (e.g.
    `gom.api.extensions.actuals`) resolve to themselves instead of their
    parent. Falls back to a depth-based heuristic for FQNs that don't appear
    under a scanned module header (e.g. `gom.Resource`, which is introduced
    via an `eval-rst` block rather than a module-style section heading).
    """
    if _KNOWN_MODULES:
        parts = fqn.split(".")
        for n in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:n])
            if candidate in _KNOWN_MODULES:
                return candidate
    if fqn.startswith("gom.api."):
        return "gom.api." + fqn[len("gom.api."):].split(".", 1)[0]
    if fqn.startswith("gom."):
        return ".".join(fqn.split(".")[:2])
    return ".".join(fqn.split(".")[:-1])


def _extract_class_fqn(fqn: str, module: str) -> str:
    """Return the class FQN between module and leaf (empty if no class segment).

    Works for both `module.Class.method` and `module.Class.Nested.method` —
    the class FQN is everything from the module boundary up to (but not
    including) the leaf, provided the first segment after the module is
    CapCase. If the first segment is lowercase, the leaf is a module-level
    function.
    """
    if not fqn.startswith(module + "."):
        return ""
    remainder = fqn[len(module) + 1:].split(".")
    if len(remainder) <= 1:
        return ""  # fqn IS module.function
    if not remainder[0][:1].isupper():
        return ""
    return module + "." + ".".join(remainder[:-1])


# =============================================================================
# MyST {py:function} parser (python_api.md)
# =============================================================================

PY_FUNC_BLOCK_RE = re.compile(
    r"^```\{py:function\}\s+(?P<sig>[^\n]+)\n"
    r"(?P<body>.*?)"
    r"^```\s*$",
    re.MULTILINE | re.DOTALL,
)


def parse_myst_python_api(text: str, source_file: str) -> list[ApiFunction]:
    blocks = list(PY_FUNC_BLOCK_RE.finditer(text))
    out: list[ApiFunction] = []

    for i, m in enumerate(blocks):
        sig_m = SIG_RE.match(m.group("sig").strip())
        if not sig_m:
            continue

        fqn = sig_m.group("fqn")
        params_raw = sig_m.group("params")
        ret_raw = (sig_m.group("ret") or "").strip()

        brief, field_text = _split_brief_and_fields(m.group("body"))
        fields = _organize_fields(_parse_fields(field_text))

        # Prose after the fence, up to the next section header
        tail_start = m.end()
        tail_end = blocks[i + 1].start() if i + 1 < len(blocks) else len(text)
        tail = text[tail_start:tail_end]
        hdg_m = re.search(r"^#{1,4}\s", tail, re.MULTILINE)
        extended = tail[:hdg_m.start() if hdg_m else len(tail)].strip()

        # If brief was empty (fence had only field list), promote from extended
        brief, extended = _promote_brief(brief, extended)

        module = _module_of(fqn)
        class_fqn = _extract_class_fqn(fqn, module)
        class_name = class_fqn.split(".")[-1] if class_fqn else ""
        merged = _merge_params(_parse_sig_params(params_raw), fields["params"])

        out.append(ApiFunction(
            fqn=fqn, module=module, name=fqn.rsplit(".", 1)[-1],
            kind="method" if class_fqn else "function",
            class_name=class_name,
            class_fqn=class_fqn,
            signature=f"{fqn}({params_raw})" + (f": {ret_raw}" if ret_raw else ""),
            params=merged,
            returns=fields["returns"],
            return_type=fields["return_type"] or ret_raw,
            description=brief,
            extended_description=extended,
            api_version=fields["api_version"],
            source_file=source_file,
            source_anchor=fqn.replace(".", "-"),
        ))
    return out


# =============================================================================
# RST {eval-rst} parser (resource_api.md)
# =============================================================================

EVAL_RST_RE = re.compile(
    r"^```\{eval-rst\}\n(?P<content>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)

RST_DIRECTIVE_RE = re.compile(
    r"^(?P<indent>[ \t]*)\.\.\s+py:(?P<kind>class|method|staticmethod|classmethod|function|attribute)::"
    r"\s+(?P<sig>[^\n]+)$",
    re.MULTILINE,
)


def parse_eval_rst_api(text: str, source_file: str,
                       classes_out: dict[str, ApiClass]) -> list[ApiFunction]:
    out: list[ApiFunction] = []
    for block in EVAL_RST_RE.finditer(text):
        out.extend(_parse_rst_block(block.group("content"), source_file, classes_out))
    return out


def _parse_rst_block(content: str, source_file: str,
                     classes_out: dict[str, ApiClass]) -> list[ApiFunction]:
    directives = list(RST_DIRECTIVE_RE.finditer(content))
    if not directives:
        return []

    current_class_fqn = ""
    out: list[ApiFunction] = []

    for i, m in enumerate(directives):
        kind = m.group("kind")
        sig = m.group("sig").strip()
        indent = m.group("indent")

        body_start = m.end()
        body_end = directives[i + 1].start() if i + 1 < len(directives) else len(content)
        body = content[body_start:body_end]

        # Dedent body by (indent + 3 spaces)
        child_indent = indent + "   "
        dedented_lines = []
        for line in body.split("\n"):
            if line.startswith(child_indent):
                dedented_lines.append(line[len(child_indent):])
            elif not line.strip():
                dedented_lines.append("")
            else:
                dedented_lines.append(line.strip())
        dedented = "\n".join(dedented_lines)

        if kind == "class":
            current_class_fqn = sig.split("(")[0].strip()
            brief, _ = _split_brief_and_fields(dedented)
            # Everything else after the first field or blank line is extended
            parts = dedented.split("\n\n", 1)
            ext = parts[1].strip() if len(parts) > 1 else ""
            name = current_class_fqn.rsplit(".", 1)[-1]
            module = _module_of(current_class_fqn + ".__placeholder")
            classes_out[current_class_fqn] = ApiClass(
                fqn=current_class_fqn, module=module, name=name,
                description=brief, extended_description=ext,
                source_file=source_file,
            )
            continue

        meth_m = re.match(r"^(?:(?P<prefix>[\w\.]+)\.)?(?P<n>\w+)\s*\((?P<params>[^)]*)\)", sig)
        if not meth_m:
            continue
        name = meth_m.group("n")
        params_raw = meth_m.group("params") or ""

        if current_class_fqn:
            fqn = f"{current_class_fqn}.{name}"
        elif meth_m.group("prefix"):
            prefix = meth_m.group("prefix")
            fqn = f"{prefix}.{name}" if "." in prefix else f"gom.{prefix}.{name}"
        else:
            fqn = name

        brief, field_text = _split_brief_and_fields(dedented)
        fields = _organize_fields(_parse_fields(field_text))
        merged = _merge_params(_parse_sig_params(params_raw), fields["params"])

        module = _module_of(fqn)
        kind_out = ("staticmethod" if kind == "staticmethod"
                    else "classmethod" if kind == "classmethod"
                    else "method" if current_class_fqn
                    else "function")

        class_fqn = current_class_fqn
        class_name = class_fqn.rsplit(".", 1)[-1] if class_fqn else ""

        out.append(ApiFunction(
            fqn=fqn, module=module, name=name,
            kind=kind_out,
            class_name=class_name,
            class_fqn=class_fqn,
            signature=f"{fqn}({params_raw})",
            params=merged,
            returns=fields["returns"],
            return_type=fields["return_type"],
            description=brief,
            api_version=fields["api_version"],
            source_file=source_file,
            source_anchor=fqn.replace(".", "-"),
        ))
    return out


# =============================================================================
# H3/H4 class scanner (catches classes even without fenced methods)
# =============================================================================

# Class header whose final segment is CapCase (= class or nested class).
# Handles gom.api.x.Class, gom.api.x.y.Class (H4 under submodule), and
# nested classes such as ScriptedCanvas.Event (H5 under H4 class). Accepts
# H3-H5 depth and one or more trailing CapCase segments:
# - 2025 style:   ### gom.api.x.ClassName
# - 2026 style:  #### gom.api.x.y.ClassName  (inside submodule)
# - nested:     ##### gom.api.x.y.ClassName.Event
CLASS_H3_RE = re.compile(
    r"^(?P<depth>#{3,5})\s+(?P<fqn>gom(?:\.[a-z_][a-z0-9_]*)+(?:\.[A-Z][\w]*)+)\s*$",
    re.MULTILINE,
)

# Stop class-body scanning at any heading (up to H6) or MyST py:* fence.
# Using {1,6} means ##### method headers correctly terminate the class body.
CLASS_BODY_STOP = re.compile(r"^(?:#{1,6}\s|```\{py:)", re.MULTILINE)


def scan_h3_classes(text: str, source_file: str) -> dict[str, ApiClass]:
    out: dict[str, ApiClass] = {}
    matches = list(CLASS_H3_RE.finditer(text))
    for i, m in enumerate(matches):
        fqn = m.group("fqn")
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]

        stop = CLASS_BODY_STOP.search(body)
        desc_chunk = body[:stop.start() if stop else len(body)].strip()

        # First paragraph -> description; rest -> extended_description
        parts = desc_chunk.split("\n\n", 1)
        brief = parts[0].strip()
        ext = parts[1].strip() if len(parts) > 1 else ""

        name = fqn.rsplit(".", 1)[-1]
        module = _module_of(fqn + ".__placeholder")

        out[fqn] = ApiClass(
            fqn=fqn, module=module, name=name,
            description=brief, extended_description=ext,
            source_file=source_file,
        )
    return out


# =============================================================================
# Module description extraction (H2-H4, matches whole-lowercase gom.* paths)
# =============================================================================

# Matches section headers for modules and submodules — all path segments
# after `gom.` are lowercase (Python convention), which is what distinguishes
# these from class headers that end in CapCase segments.
MODULE_HEADER_RE = re.compile(
    r"^(?P<depth>#{2,4})\s+(?P<n>gom(?:\.[a-z_][a-z0-9_]*)+)\s*$",
    re.MULTILINE,
)


def _scan_known_modules(text: str) -> set[str]:
    """Collect module/submodule names from section headers in one doc page.

    H2 `## gom.x` is always a module. H3 `### gom.x.y` with a lowercase final
    segment is ambiguous — it could be a submodule (e.g.
    `gom.api.extensions.actuals`) or a module-level function (e.g.
    `gom.api.addons.get_addon`). We disambiguate by requiring at least one H4
    child heading whose FQN extends the H3's path; that's the structural
    signature of a submodule. H4+ lowercase headers are always functions.
    """
    out: set[str] = set()
    matches = list(MODULE_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        depth = len(m.group("depth"))
        name = m.group("n").strip()
        if depth == 2:
            out.add(name)
            continue
        if depth >= 4:
            continue
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():body_end]
        if re.search(rf"^####\s+{re.escape(name)}\.", body, re.MULTILINE):
            out.add(name)
    return out


def parse_module_descriptions(text: str) -> dict[str, str]:
    """Extract per-module prose that precedes the first subheading/fence.

    Only headers that pass `_scan_known_modules` get a description; this
    keeps function-definition headers (which share the `### gom.x.y` shape)
    from being mistakenly treated as modules.
    """
    valid = _scan_known_modules(text)
    out: dict[str, str] = {}
    matches = list(MODULE_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group("n").strip()
        if name not in valid:
            continue
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        stop = re.search(r"^(?:#{2,6}\s|```)", body, re.MULTILINE)
        desc = body[:stop.start() if stop else len(body)].strip()
        out[name] = desc
    return out


# =============================================================================
# API collection dispatcher
# =============================================================================

def collect_api(api_repo: Path) -> tuple[
    dict[str, ApiFunction], dict[str, ApiClass], dict[str, ModuleMeta]
]:
    api_dir = api_repo / "doc" / "python_api"
    if not api_dir.is_dir():
        print(f"  warning: {api_dir} not found", file=sys.stderr)
        return {}, {}, {}

    # Pre-pass: discover every module/submodule header across all doc pages so
    # subsequent FQN resolution can prefer longest-prefix matches. Without
    # this, a method like `gom.api.extensions.sequence.ScriptedSequenceElement.create`
    # gets its module resolved to `gom.api.extensions` and then its class_fqn
    # extraction bails out because the first between-segment is lowercase.
    _KNOWN_MODULES.clear()
    docs: list[tuple[Path, str, str]] = []
    for md in sorted(api_dir.glob("*.md")):
        text = _normalize(md.read_text(encoding="utf-8"))
        rel = str(md.relative_to(api_repo))
        docs.append((md, text, rel))
        _KNOWN_MODULES.update(_scan_known_modules(text))

    functions: dict[str, ApiFunction] = {}
    classes: dict[str, ApiClass] = {}
    module_descs: dict[str, str] = {}

    for md, text, rel in docs:
        myst_fns = parse_myst_python_api(text, rel)
        rst_fns = parse_eval_rst_api(text, rel, classes)  # populates classes for RST case
        h3_classes = scan_h3_classes(text, rel)            # MyST H3-H5 classes incl. nested

        # H3 classes are canonical (have descriptions); only add if not already set
        for fqn, cls in h3_classes.items():
            if fqn not in classes:
                classes[fqn] = cls
            else:
                # Prefer H3 description/extended if RST version was stub-only
                existing = classes[fqn]
                if not existing.description and cls.description:
                    existing.description = cls.description
                if not existing.extended_description and cls.extended_description:
                    existing.extended_description = cls.extended_description

        fns = myst_fns + rst_fns
        print(f"    {md.name}: {len(myst_fns)} MyST + {len(rst_fns)} RST fns, "
              f"+{len(h3_classes)} H3 classes", file=sys.stderr)
        for fn in fns:
            functions[fn.fqn] = fn

        module_descs.update(parse_module_descriptions(text))

    # Synthesize stub class entries for any class_fqn referenced by a method
    # but never registered from a section header. This catches docs like the
    # `views` submodule where `ScriptedEditor` and `ScriptedView` have no
    # explicit class heading of their own — only their methods appear.
    for fn in functions.values():
        if fn.class_fqn and fn.class_fqn not in classes:
            classes[fn.class_fqn] = ApiClass(
                fqn=fn.class_fqn,
                module=_module_of(fn.class_fqn + ".__placeholder"),
                name=fn.class_fqn.rsplit(".", 1)[-1],
                source_file=fn.source_file,
            )

    # Wire class methods
    for fn in functions.values():
        if fn.class_fqn and fn.class_fqn in classes:
            classes[fn.class_fqn].methods.append(fn.fqn)
    for cls in classes.values():
        cls.methods.sort()

    # Build ModuleMeta index. Seed with every scanned module header so
    # submodules like `gom.api.extensions.actuals` always show up as
    # navigable units even when empty.
    modules: dict[str, ModuleMeta] = {
        name: ModuleMeta(name=name) for name in _KNOWN_MODULES
    }
    for fqn, fn in functions.items():
        mm = modules.setdefault(fn.module, ModuleMeta(name=fn.module))
        if fn.class_fqn:
            if fn.class_fqn not in mm.classes:
                mm.classes.append(fn.class_fqn)
        else:
            mm.functions.append(fqn)
    for cls_fqn, cls in classes.items():
        mm = modules.setdefault(cls.module, ModuleMeta(name=cls.module))
        if cls_fqn not in mm.classes:
            mm.classes.append(cls_fqn)

    for name, desc in module_descs.items():
        mm = modules.setdefault(name, ModuleMeta(name=name))
        mm.description = desc

    for mm in modules.values():
        mm.functions.sort()
        mm.classes.sort()

    return functions, classes, modules


# =============================================================================
# HowTo parser
# =============================================================================

def parse_howto(md_path: Path, repo_root: Path, howtos_root: Path) -> HowTo:
    text = _normalize(md_path.read_text(encoding="utf-8"))
    rel = md_path.relative_to(repo_root)

    stripped = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
    title_m = re.search(r"^#\s+(.+?)$", stripped, re.MULTILINE)
    title = title_m.group(1).strip() if title_m else md_path.stem

    mentions = re.findall(r"\bgom\.(?:api|Resource|app|script|interactive)[\w\.]*", text)
    api_mentions = sorted({m.rstrip(".,;:)]}'\"`") for m in mentions})

    linked = sorted({
        Path(m.group(1)).stem
        for m in re.finditer(r"\]\(([^)]+?\.md)[^)]*\)", text)
    })

    rel_to_ht = md_path.relative_to(howtos_root).with_suffix("")
    slug = ".".join(rel_to_ht.parts) if rel_to_ht.parts else md_path.stem

    return HowTo(
        slug=slug, title=title, source_file=str(rel), content=text,
        api_mentions=api_mentions, linked_howtos=linked,
    )


def collect_howtos(api_repo: Path) -> dict[str, HowTo]:
    howtos_dir = api_repo / "doc" / "howtos"
    out: dict[str, HowTo] = {}
    if howtos_dir.is_dir():
        for md in sorted(howtos_dir.rglob("*.md")):
            if "assets" in md.parts:
                continue
            ht = parse_howto(md, api_repo, howtos_dir)
            out[ht.slug] = ht

    # The deprecated scripted_elements_api.md page doesn't match any of the
    # API parser formats (no {py:function}/{eval-rst} fences, no H2 gom.*
    # module headings — it documents user-defined callbacks like `dialog()`
    # and `calculation()`). Surface it as a howto so the content is
    # retrievable instead of silently dropped.
    legacy = api_repo / "doc" / "python_api" / "scripted_elements_api.md"
    if legacy.is_file():
        text = _normalize(legacy.read_text(encoding="utf-8"))
        title_m = re.search(r"^#\s+(.+?)$", text, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else legacy.stem
        mentions = re.findall(r"\bgom\.(?:api|Resource|app|script|interactive)[\w\.]*", text)
        api_mentions = sorted({m.rstrip(".,;:)]}'\"`") for m in mentions})
        linked = sorted({
            Path(m.group(1)).stem
            for m in re.finditer(r"\]\(([^)]+?\.md)[^)]*\)", text)
        })
        slug = "scripted_elements_api"
        out[slug] = HowTo(
            slug=slug, title=title,
            source_file=str(legacy.relative_to(api_repo)),
            content=text,
            api_mentions=api_mentions,
            linked_howtos=linked,
        )

    return out


# =============================================================================
# Example parser
# =============================================================================

def _ex_field_re(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"^{re.escape(name)}:\s*\n:\s*(.+?)(?=\n\S[\w ]*:\s*\n:|\n##|\Z)",
        re.DOTALL | re.MULTILINE,
    )

DESC_RE = _ex_field_re("Description")
TAGS_RE = _ex_field_re("Tags")
REFS_RE = _ex_field_re("References")
VERSION_EX_RE = _ex_field_re("Required Software")
PROJECTS_RE = _ex_field_re("Example Projects")

LINK_RE = re.compile(r"\[(\w+)\]\(([^)]+)\)")
TAG_BADGE_RE = re.compile(r"badge/([\w-]+)-blue")


def parse_example_dir(ex_dir: Path, ex_repo: Path) -> Example | None:
    doc_md = ex_dir / "doc" / "Documentation.md"
    if not doc_md.exists():
        return None

    text = _normalize(doc_md.read_text(encoding="utf-8"))

    def _grab(r: re.Pattern[str]) -> str:
        m = r.search(text)
        return m.group(1).strip() if m else ""

    projects_raw = _grab(PROJECTS_RE)
    example_projects = [m.group(1) for m in re.finditer(r"\[([^\]]+)\]", projects_raw)]

    tags = [s.replace("--", "-") for s in TAG_BADGE_RE.findall(_grab(TAGS_RE))]

    api_refs: list[str] = []
    howto_refs: list[str] = []
    for label, url in LINK_RE.findall(_grab(REFS_RE)):
        (api_refs if label.lower() == "api" else howto_refs).append(url)

    scripts: dict[str, str] = {}
    scripts_dir = ex_dir / "scripts"
    if scripts_dir.is_dir():
        for py in scripts_dir.rglob("*.py"):
            scripts[str(py.relative_to(ex_dir))] = py.read_text(encoding="utf-8")

    api_calls = sorted({c for src in scripts.values() for c in extract_gom_calls(src)})

    return Example(
        name=ex_dir.name, category=ex_dir.parent.name,
        path=str(ex_dir.relative_to(ex_repo)),
        description=_grab(DESC_RE),
        tags=tags,
        version_min=_grab(VERSION_EX_RE),
        api_refs=api_refs, howto_refs=howto_refs,
        api_calls=api_calls,
        documentation=text,
        scripts=scripts,
        example_projects=example_projects,
    )


def collect_examples(ex_repo: Path) -> dict[str, Example]:
    root = ex_repo / "AppExamples"
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found")
    out: dict[str, Example] = {}
    for cat in sorted(p for p in root.iterdir() if p.is_dir()):
        for ex in sorted(p for p in cat.iterdir() if p.is_dir()):
            parsed = parse_example_dir(ex, ex_repo)
            if parsed:
                out[parsed.name] = parsed
    return out


# =============================================================================
# AST: extract maximal gom.* attribute chains from Python source
# =============================================================================

def extract_gom_calls(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    chains: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _unparse_attr_chain(node)
            if chain.startswith("gom.") and chain != "gom":
                chains.add(chain)

    maximal: set[str] = set()
    for c in sorted(chains, key=len, reverse=True):
        if not any(other.startswith(c + ".") for other in maximal):
            maximal.add(c)
    return maximal


def _unparse_attr_chain(node: ast.AST) -> str:
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


# =============================================================================
# Cross-linking
# =============================================================================

def crosslink(api: dict[str, ApiFunction], classes: dict[str, ApiClass],
              modules: dict[str, ModuleMeta],
              examples: dict[str, Example], howtos: dict[str, HowTo]) -> None:
    """Resolve every gom.* mention against functions, classes, AND modules.

    A mention can be:
      - an exact function fqn  (gom.api.settings.get)
      - an exact class fqn     (gom.api.addons.AddOn, gom.Resource)
      - a module name          (gom.api.imaging)
      - something undocumented (gom.Vec3d) -> skipped
    """
    def _resolve(mention: str) -> tuple[str, str] | None:
        """Return (node_type, fqn) or None."""
        if mention in api:
            return ("function", mention)
        if mention in classes:
            return ("class", mention)
        if mention in modules:
            return ("module", mention)
        return None

    # Example -> API (via AST-extracted gom.* chains)
    for ex in examples.values():
        for call in ex.api_calls:
            hit = _resolve(call)
            if hit is None:
                # Try progressively shorter prefixes so `gom.api.addons.AddOn.exists(...)`
                # extracted as `gom.api.addons.AddOn.exists` also counts toward the class
                # and the module.
                parts = call.split(".")
                for n in range(len(parts) - 1, 1, -1):
                    hit = _resolve(".".join(parts[:n]))
                    if hit:
                        break
            if hit is None:
                continue
            node_type, fqn = hit
            if node_type == "function":
                api[fqn].used_by_examples.append(ex.name)
            elif node_type == "class":
                classes[fqn].used_by_examples.append(ex.name)
            elif node_type == "module":
                modules[fqn].used_by_examples.append(ex.name)

    # HowTo -> API (via regex mentions in prose)
    for ht in howtos.values():
        for mention in ht.api_mentions:
            hit = _resolve(mention)
            if hit is None:
                parts = mention.split(".")
                for n in range(len(parts) - 1, 1, -1):
                    hit = _resolve(".".join(parts[:n]))
                    if hit:
                        break
            if hit is None:
                continue
            node_type, fqn = hit
            if node_type == "function":
                api[fqn].mentioned_in_howtos.append(ht.slug)
            elif node_type == "class":
                classes[fqn].mentioned_in_howtos.append(ht.slug)
            elif node_type == "module":
                modules[fqn].mentioned_in_howtos.append(ht.slug)

    # HowTo -> Example (name match in prose)
    for ht in howtos.values():
        found: list[str] = []
        for ex_name in examples:
            if re.search(rf"\b{re.escape(ex_name)}\b", ht.content):
                found.append(ex_name)
                examples[ex_name].mentioned_in_howtos.append(ht.slug)
        ht.example_mentions = sorted(set(found))

    # Dedup + sort everywhere
    for fn in api.values():
        fn.used_by_examples = sorted(set(fn.used_by_examples))
        fn.mentioned_in_howtos = sorted(set(fn.mentioned_in_howtos))
    for cls in classes.values():
        cls.used_by_examples = sorted(set(cls.used_by_examples))
        cls.mentioned_in_howtos = sorted(set(cls.mentioned_in_howtos))
    for mm in modules.values():
        mm.used_by_examples = sorted(set(mm.used_by_examples))
        mm.mentioned_in_howtos = sorted(set(mm.mentioned_in_howtos))
    for ex in examples.values():
        ex.mentioned_in_howtos = sorted(set(ex.mentioned_in_howtos))


# =============================================================================
# Main
# =============================================================================

def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zeiss-version", default="2026", metavar="YEAR",
                    help="ZEISS INSPECT version to target; must match a release "
                         "branch in the api repo (default: 2026)")
    ap.add_argument("--workspace", type=Path, default=Path("repos"),
                    help="where to clone the two ZEISS repos (default: ./repos)")
    ap.add_argument("--api-repo", type=Path, default=None,
                    help="override: use an existing local clone of "
                         "zeiss-inspect-app-api instead of auto-cloning")
    ap.add_argument("--ex-repo", type=Path, default=None,
                    help="override: use an existing local clone of "
                         "zeiss-inspect-app-examples instead of auto-cloning")
    ap.add_argument("--ex-branch", default=None, metavar="BRANCH",
                    help="examples-repo branch to clone/checkout. Bypasses "
                         "year-based detection. Useful when examples for a "
                         "new version live on a feature branch that hasn't "
                         "been merged to main yet "
                         "(e.g. 20251211-feat-custom-elements).")
    ap.add_argument("--out", type=Path, default=Path("corpus"),
                    help="corpus output directory (default: ./corpus)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip `git fetch`/`pull` on existing clones (offline mode)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[0/4] Preparing repos for ZEISS INSPECT {args.zeiss_version}", file=sys.stderr)
    api_repo, ex_repo, meta = prepare_repos(
        workspace=args.workspace,
        zeiss_version=args.zeiss_version,
        fetch=not args.no_fetch,
        api_override=args.api_repo,
        ex_override=args.ex_repo,
        ex_branch_override=args.ex_branch,
    )
    print(f"      api repo: {api_repo} @ {meta['api_repo']['branch']} "
          f"({meta['api_repo']['commit']})", file=sys.stderr)
    print(f"      ex  repo: {ex_repo} @ {meta['ex_repo']['branch']} "
          f"({meta['ex_repo']['commit']})", file=sys.stderr)

    print(f"[1/4] API docs <- {api_repo}/doc/python_api", file=sys.stderr)
    api, classes, modules = collect_api(api_repo)
    brief_ok = sum(1 for f in api.values() if f.description)
    print(f"      {len(api)} functions / {len(classes)} classes / {len(modules)} modules",
          file=sys.stderr)
    print(f"      {brief_ok}/{len(api)} functions have a description",
          file=sys.stderr)

    print(f"[2/4] How-tos <- {api_repo}/doc/howtos", file=sys.stderr)
    howtos = collect_howtos(api_repo)
    print(f"      {len(howtos)} guides parsed", file=sys.stderr)

    print(f"[3/4] Examples <- {ex_repo}/AppExamples", file=sys.stderr)
    examples = collect_examples(ex_repo)
    print(f"      {len(examples)} examples parsed", file=sys.stderr)

    print("[4/4] Cross-linking", file=sys.stderr)
    crosslink(api, classes, modules, examples, howtos)

    linked_ex = sum(1 for e in examples.values() if any(
        c in api or c in classes or c in modules or
        any(".".join(c.split(".")[:n]) in api or ".".join(c.split(".")[:n]) in classes
            or ".".join(c.split(".")[:n]) in modules for n in range(len(c.split(".")), 1, -1))
        for c in e.api_calls
    ))
    covered_fn_ex = sum(1 for f in api.values() if f.used_by_examples)
    covered_fn_ht = sum(1 for f in api.values() if f.mentioned_in_howtos)
    covered_cls_ex = sum(1 for c in classes.values() if c.used_by_examples)
    covered_cls_ht = sum(1 for c in classes.values() if c.mentioned_in_howtos)
    covered_mod_ex = sum(1 for m in modules.values() if m.used_by_examples)
    covered_mod_ht = sum(1 for m in modules.values() if m.mentioned_in_howtos)
    print(f"      examples linked to corpus: {linked_ex}/{len(examples)}", file=sys.stderr)
    print(f"      functions covered: {covered_fn_ex} by examples, {covered_fn_ht} by howtos (/{len(api)})",
          file=sys.stderr)
    print(f"      classes covered:   {covered_cls_ex} by examples, {covered_cls_ht} by howtos (/{len(classes)})",
          file=sys.stderr)
    print(f"      modules covered:   {covered_mod_ex} by examples, {covered_mod_ht} by howtos (/{len(modules)})",
          file=sys.stderr)

    all_calls: set[str] = set()
    for ex in examples.values():
        all_calls.update(ex.api_calls)
    def _resolves(c: str) -> bool:
        if c in api or c in classes or c in modules:
            return True
        parts = c.split(".")
        for n in range(len(parts) - 1, 1, -1):
            p = ".".join(parts[:n])
            if p in api or p in classes or p in modules:
                return True
        return False
    unmatched = sorted(c for c in all_calls if not _resolves(c))
    if unmatched:
        print(f"      unmatched gom.* chains (undocumented types): {len(unmatched)}",
              file=sys.stderr)
        for c in unmatched[:8]:
            print(f"        {c}", file=sys.stderr)
        if len(unmatched) > 8:
            print(f"        ... and {len(unmatched) - 8} more", file=sys.stderr)

    _dump(args.out / "api_functions.json",
          {k: asdict(v) for k, v in sorted(api.items())})
    _dump(args.out / "api_classes.json",
          {k: asdict(v) for k, v in sorted(classes.items())})
    _dump(args.out / "modules.json",
          {k: asdict(v) for k, v in sorted(modules.items())})
    _dump(args.out / "howtos.json",
          {k: asdict(v) for k, v in sorted(howtos.items())})
    _dump(args.out / "examples.json",
          {k: asdict(v) for k, v in sorted(examples.items())})
    _dump(args.out / "corpus_meta.json", meta)

    print(f"\nWrote {args.out}/ (INSPECT {args.zeiss_version}, "
          f"api@{meta['api_repo']['commit']}, ex@{meta['ex_repo']['commit']})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
