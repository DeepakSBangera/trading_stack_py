# scripts/w20_freeze_guard.py
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(r"F:\Projects\trading_stack_py")
CONFIG = ROOT / "config" / "change_control.yaml"
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "w20_freeze_status.json"


def _read_yaml_like(path: Path) -> dict:
    """
    Very small, tolerant YAML-ish reader sufficient for our change_control.yaml.
    Supports:
      - key: value
      - key:
          - item
          - item2
      - multiline scalar after `key: ""` or `key: |` (treated as raw text)
    Auto-converts a previously-seen scalar to list if we encounter `- item`.
    """
    out: dict = {}
    if not path.exists():
        return out

    text = path.read_text(encoding="utf-8", errors="ignore")
    current_key: str | None = None
    collecting_multiline = False

    for raw in text.splitlines():
        line = raw.rstrip("\n")

        # comments or blank
        if not line.strip() or line.strip().startswith("#"):
            continue

        # list item?
        m_list = re.match(r"^\s*-\s*(.+)$", line)
        if m_list and current_key:
            item = m_list.group(1).strip()
            # ensure list
            if not isinstance(out.get(current_key), list):
                prev = out.get(current_key, None)
                out[current_key] = []
                # if previous scalar was non-empty and not a boolean, keep it as first list element
                if isinstance(prev, str) and prev not in ("",):
                    out[current_key].append(prev)
            out[current_key].append(item)
            collecting_multiline = False
            continue

        # key: value
        m_kv = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$", line)
        if m_kv:
            key = m_kv.group(1).strip()
            val = m_kv.group(2).strip()

            current_key = key
            collecting_multiline = False

            # explicit list opener (empty value; list will follow)
            if val == "" or val == "|":
                # start empty scalar if '|' (multiline), empty string otherwise; converted later if we see '-'
                out[key] = "" if val == "|" else ""
                if val == "|":
                    collecting_multiline = True
                continue

            # booleans
            low = val.lower()
            if low in ("true", "false"):
                out[key] = low == "true"
                continue

            # quoted string stripping
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]

            out[key] = val
            continue

        # multiline scalar continuation
        if collecting_multiline and current_key:
            # append line to scalar (with newline separation)
            prev = out.get(current_key, "")
            out[current_key] = prev + ("\n" if prev else "") + line
            continue

        # if we reach here, ignore unknown formatting
        # (keeps parser tolerant for YAML we don't use)

    return out


def _git_branch(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    if not head.exists():
        return None
    s = head.read_text(errors="ignore").strip()
    m = re.search(r"refs/heads/(.+)$", s)
    if m:
        return m.group(1).strip()
    return "DETACHED"


def _git_sha_short(repo: Path) -> str | None:
    last = REPORTS / "run_manifest_last.json"
    if last.exists():
        try:
            j = json.loads(last.read_text(encoding="utf-8"))
            sha = j.get("git_sha8") or j.get("git_sha") or ""
            return sha[:8] if sha else None
        except Exception:
            pass
    br = _git_branch(repo)
    if br and br != "DETACHED":
        ref = repo / ".git" / "refs" / "heads" / br
        if ref.exists():
            full = ref.read_text(errors="ignore").strip()
            return full[:8] if full else None
    return None


def _now_ist_iso() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).isoformat(
        timespec="seconds"
    )


def _within_window(now: dt.datetime, start: str | None, end: str | None) -> bool:
    def parse(s: str | None):
        if not s:
            return None
        try:
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    s = parse(start)
    e = parse(end)
    if s and now < s:
        return False
    if e and now > e:
        return False
    return True


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    cfg = _read_yaml_like(CONFIG)

    freeze_on = bool(cfg.get("freeze_on", False))
    reason = cfg.get("reason", "")
    eff_from = cfg.get("effective_from")
    eff_to = cfg.get("effective_to")
    allowed = cfg.get("allowed_branches", [])
    blocked_pat = cfg.get("blocked_scripts", [])

    # Normalize list-y keys
    if not isinstance(allowed, list):
        allowed = [allowed] if allowed else []
    if not isinstance(blocked_pat, list):
        blocked_pat = [blocked_pat] if blocked_pat else []

    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    in_window = _within_window(now, eff_from, eff_to) if (eff_from or eff_to) else True

    branch = _git_branch(ROOT) or "UNKNOWN"
    sha8 = _git_sha_short(ROOT) or "????????"

    # match helpers
    def _globmatch(pat: str, text: str) -> bool:
        return re.fullmatch(pat.replace("*", ".*"), text) is not None

    branch_allowed = any(_globmatch(p, branch) for p in allowed) if allowed else True
    freeze_active = bool(freeze_on and in_window and not branch_allowed)

    status = {
        "as_of_ist": _now_ist_iso(),
        "freeze_on": freeze_on,
        "in_window": in_window,
        "effective_from": eff_from,
        "effective_to": eff_to,
        "reason": reason,
        "branch": branch,
        "git_sha8": sha8,
        "allowed_branches": allowed,
        "blocked_patterns": blocked_pat,
        "branch_allowed": branch_allowed,
        "freeze_active": freeze_active,
    }

    OUT_JSON.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
