"""Reader for the official Claude Code statusline ``rate_limits`` (trust keystone).

Claude Code (v2.1.80+, Pro/Max) pipes a session JSON to the statusline command
on stdin; that JSON carries ``rate_limits.{five_hour,seven_day}`` with an
official ``used_percentage`` (0-100) and ``resets_at`` (Unix epoch seconds). The
``--statusline`` writer captures that block to ``~/.claude-monitor/statusline/
latest.json``; this module reads it back so the snapshot can label the official
limit as ``confidence="official"`` instead of a local estimate.

No network. Defensive against the known leak bug (#52326) where ``used_percentage``
can carry the ``resets_at`` epoch instead of a real percentage.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _finite_int(value: Any) -> Optional[int]:
    """Coerce a finite real epoch to int, else None.

    JSON permits ``NaN``/``Infinity`` (Python's ``json`` accepts them by default),
    which would crash ``int()`` or poison comparisons; reject them and bools.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return int(value)


# How long an official capture stays "fresh". Claude Code refreshes the
# statusline frequently while a session is active; if our capture is older than
# this the official numbers may lag, so we flag the snapshot stale.
OFFICIAL_TTL_SECONDS = 600


def default_statusline_path() -> Path:
    """Where the ``--statusline`` writer drops the latest official capture."""
    return Path.home() / ".claude-monitor" / "statusline" / "latest.json"


def _clean_pct(value: Any) -> Optional[float]:
    """Sanitize an official ``used_percentage`` (0-100), guarding leak bug #52326.

    A value just over 100 is a rounding artifact (clamp to 100); an epoch-sized
    value is the leaked ``resets_at`` and is dropped to ``None`` rather than shown
    as a bogus 100%.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(value):  # JSON NaN/Infinity must never render as a percent
        return None
    if value < 0:
        return None
    if value > 100:
        return 100.0 if value <= 101 else None
    return float(value)


def _window(raw: Any, now_epoch: Optional[int]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    epoch = _finite_int(raw.get("resets_at"))
    pct = _clean_pct(raw.get("used_percentage"))
    # At or past the reset time the window has rolled over: the captured
    # percentage is for an expired window and no longer reflects the live limit.
    if epoch is not None and now_epoch is not None and now_epoch >= epoch:
        pct = None
    return {"used_percentage": pct, "resets_at_epoch": epoch}


def read_official_limits(
    path: Optional[Path] = None, now_epoch: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Read the captured official limits, or ``None`` if unavailable/unusable.

    Returns ``{five_hour, seven_day, captured_at_epoch, stale}`` where each window
    is ``{used_percentage, resets_at_epoch}`` or ``None`` when that window is absent.
    """
    path = path or default_statusline_path()
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    rate_limits = payload.get("rate_limits")
    if not isinstance(rate_limits, dict):
        return None

    captured = _finite_int(payload.get("captured_at_epoch"))
    stale = (
        now_epoch is not None
        and captured is not None
        and (now_epoch - captured) > OFFICIAL_TTL_SECONDS
    )

    return {
        "five_hour": _window(rate_limits.get("five_hour"), now_epoch),
        "seven_day": _window(rate_limits.get("seven_day"), now_epoch),
        "captured_at_epoch": captured,
        "stale": bool(stale),
    }


# Claude Code caches the account's live utilization in its own config file
# (``~/.claude.json`` -> ``cachedUsageUtilization``) and refreshes it whenever it
# fetches usage (e.g. ``/usage``). It is the same server-side number the
# statusline reports, so it is an official fallback when no capture exists.
CLAUDE_CACHE_KIND = "claude_code_cache"
CLAUDE_CACHE_TTL_SECONDS = 3600


def default_claude_config_path() -> Path:
    """Claude Code's global config file (honors ``CLAUDE_CONFIG_DIR``)."""
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir).expanduser() / ".claude.json"
    return Path.home() / ".claude.json"


def _iso_epoch(value: Any) -> Optional[int]:
    if not isinstance(value, str) or not value:
        return _finite_int(value)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _cache_window(raw: Any, now_epoch: Optional[int]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    return _window(
        {
            "used_percentage": raw.get("utilization"),
            "resets_at": _iso_epoch(raw.get("resets_at")),
        },
        now_epoch,
    )


def read_claude_code_cache_limits(
    path: Optional[Path] = None, now_epoch: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Read the utilization Claude Code cached in its config file.

    Returns the same shape as :func:`read_official_limits` (plus ``source``), or
    ``None`` when the file or the cache block is missing.
    """
    path = path or default_claude_config_path()
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    cache = payload.get("cachedUsageUtilization") if isinstance(payload, dict) else None
    if not isinstance(cache, dict) or not isinstance(cache.get("utilization"), dict):
        return None
    utilization = cache["utilization"]

    fetched_ms = _finite_int(cache.get("fetchedAtMs"))
    captured = fetched_ms // 1000 if fetched_ms is not None else None
    stale = now_epoch is not None and (
        captured is None or (now_epoch - captured) > CLAUDE_CACHE_TTL_SECONDS
    )
    return {
        "five_hour": _cache_window(utilization.get("five_hour"), now_epoch),
        "seven_day": _cache_window(utilization.get("seven_day"), now_epoch),
        "captured_at_epoch": captured,
        "stale": bool(stale),
        "source": {"kind": CLAUDE_CACHE_KIND},
    }


def _has_fresh_percentage(limits: Optional[Dict[str, Any]]) -> bool:
    if not limits or limits.get("stale"):
        return False
    return any(
        isinstance(limits.get(key), dict)
        and limits[key].get("used_percentage") is not None
        for key in ("five_hour", "seven_day")
    )


def select_official_limits(
    statusline: Optional[Dict[str, Any]], claude_cache: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Pick the best official source: a fresh statusline capture, else a fresh
    Claude Code cache, else whatever (possibly stale) statusline data exists."""
    if _has_fresh_percentage(statusline):
        return statusline
    if _has_fresh_percentage(claude_cache):
        return claude_cache
    return statusline


def capture_statusline(
    stdin_payload: Dict[str, Any],
    path: Optional[Path] = None,
    now_epoch: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Persist the official ``rate_limits`` from a statusline stdin payload.

    Used by the ``--statusline`` hook: Claude Code pipes its session JSON to us,
    we stash the ``rate_limits`` block (plus a capture timestamp) so the monitor
    can read it as the official limit. Returns the written capture, or ``None``
    when the payload has no ``rate_limits`` (free tier / older Claude Code) — but
    still writes a tombstone in that case so a plan downgrade clears any prior
    official data instead of serving it forever. Writes atomically (pid-unique
    temp + ``os.replace``) so a concurrent reader never sees a half-written file.
    """
    if not isinstance(stdin_payload, dict):  # stdin may be any JSON value
        stdin_payload = {}
    rate_limits = stdin_payload.get("rate_limits")
    has_official = isinstance(rate_limits, dict) and bool(rate_limits)
    capture = {
        "captured_at_epoch": now_epoch,
        "rate_limits": rate_limits if has_official else None,
    }

    path = path or default_statusline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(capture))
    os.replace(tmp, path)
    return capture if has_official else None


def format_statusline(
    stdin_payload: Dict[str, Any], capture: Optional[Dict[str, Any]]
) -> str:
    """Render the one-line status bar shown by Claude Code's statusline.

    Shows the model name and the official 5h/7d usage we just captured. Leaked
    percentages (bug #52326) are skipped. Falls back to ``claude-monitor`` so the
    bar is never blank.
    """
    parts = []
    model = stdin_payload.get("model") if isinstance(stdin_payload, dict) else None
    name = model.get("display_name") if isinstance(model, dict) else None
    if name:
        parts.append(str(name))

    rate_limits = (capture or {}).get("rate_limits")
    if isinstance(rate_limits, dict):
        for label, key in (("5h", "five_hour"), ("7d", "seven_day")):
            window = rate_limits.get(key)
            pct = (
                _clean_pct(window.get("used_percentage"))
                if isinstance(window, dict)
                else None
            )
            if pct is not None:
                parts.append(f"{label} {pct:.0f}%")

    return " · ".join(parts) if parts else "claude-monitor"
