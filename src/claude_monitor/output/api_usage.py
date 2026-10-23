"""Opt-in reader for Anthropic's experimental OAuth usage endpoint (#202).

This endpoint is undocumented, so callers must opt in. Data from it is labeled
``confidence="experimental"`` by the snapshot builder and never outranks fresh
official statusline limits.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

API_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
API_TTL_SECONDS = 180
API_KIND = "anthropic_oauth_usage_api"
_BETA_HEADER = "oauth-2025-04-20"
_KEYCHAIN_SERVICE = "Claude Code-credentials"


def default_api_cache_path() -> Path:
    """Default cache file for experimental OAuth usage responses."""
    return Path.home() / ".claude-monitor" / "api" / "latest.json"


def _finite_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return int(value)


def _epoch(value: Any) -> Optional[int]:
    numeric = _finite_int(value)
    if numeric is not None:
        return numeric
    if not isinstance(value, str) or not value:
        return None
    try:
        iso = value.replace("Z", "+00:00") if value.endswith("Z") else value
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())


def _utilization_to_pct(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    # The endpoint reports utilization in percent (Claude Code caches the same
    # response with e.g. ``"utilization": 7``); scaling values <= 1 as fractions
    # turned a real 1% into 100%.
    pct = float(value)
    if pct > 100:
        return 100.0 if pct <= 101 else None
    return round(pct, 1)


def _window(raw: Any, now_epoch: Optional[int]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    utilization = (
        raw["utilization"] if "utilization" in raw else raw.get("used_percentage")
    )
    pct = _utilization_to_pct(utilization)
    reset_value = (
        raw.get("resets_at") if "resets_at" in raw else raw.get("resets_at_epoch")
    )
    reset_epoch = _epoch(reset_value)
    if reset_epoch is not None and now_epoch is not None and now_epoch >= reset_epoch:
        pct = None
    return {"used_percentage": pct, "resets_at_epoch": reset_epoch}


def _normalize(
    payload: Dict[str, Any],
    captured_at_epoch: int,
    now_epoch: Optional[int],
    stale: bool = False,
) -> Dict[str, Any]:
    limits = (
        payload.get("rate_limits")
        if isinstance(payload.get("rate_limits"), dict)
        else payload
    )
    return {
        "five_hour": _window(limits.get("five_hour"), now_epoch),
        "seven_day": _window(limits.get("seven_day"), now_epoch),
        "captured_at_epoch": captured_at_epoch,
        "stale": bool(stale),
        "source": {"kind": API_KIND},
    }


def _read_cache(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _cached_limits(
    cache: Optional[Dict[str, Any]],
    now_epoch: Optional[int],
    ttl_seconds: int,
) -> Optional[Dict[str, Any]]:
    if not cache or not isinstance(cache.get("limits"), dict):
        return None
    limits = copy.deepcopy(cache["limits"])
    captured = _finite_int(limits.get("captured_at_epoch")) or _finite_int(
        cache.get("captured_at_epoch")
    )
    stale = False
    if captured is not None and now_epoch is not None:
        stale = (now_epoch - captured) > ttl_seconds
    limits["five_hour"] = _window(limits.get("five_hour"), now_epoch)
    limits["seven_day"] = _window(limits.get("seven_day"), now_epoch)
    limits["captured_at_epoch"] = captured
    limits["stale"] = bool(stale)
    limits.setdefault("source", {"kind": API_KIND})
    return limits


def _cache_is_fresh(
    cache: Optional[Dict[str, Any]], now_epoch: Optional[int], ttl_seconds: int
) -> bool:
    if not cache:
        return False
    captured = _finite_int(cache.get("captured_at_epoch"))
    if captured is None and isinstance(cache.get("limits"), dict):
        captured = _finite_int(cache["limits"].get("captured_at_epoch"))
    if captured is None or now_epoch is None:
        return False
    return (now_epoch - captured) <= ttl_seconds


def _write_cache(
    path: Path,
    limits: Optional[Dict[str, Any]],
    captured_at_epoch: int,
    retry_after_epoch: Optional[int] = None,
) -> None:
    payload: Dict[str, Any] = {"captured_at_epoch": captured_at_epoch}
    if limits is not None:
        payload["limits"] = limits
    if retry_after_epoch is not None:
        payload["retry_after_epoch"] = retry_after_epoch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _token_from_payload(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in ("accessToken", "access_token", "oauth_access_token"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("claudeAiOauth", "oauth"):
        token = _token_from_payload(payload.get(key))
        if token:
            return token
    return None


def read_oauth_token(credentials_path: Optional[Path] = None) -> Optional[str]:
    """Return an OAuth access token from env or Claude's credentials file."""
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        return token

    path = credentials_path or (Path.home() / ".claude" / ".credentials.json")
    try:
        return _token_from_payload(json.loads(path.read_text()))
    except (OSError, ValueError):
        pass
    if credentials_path is None and sys.platform == "darwin":
        return _keychain_token()
    return None


def _keychain_token() -> Optional[str]:
    """Claude Code on macOS keeps its OAuth credentials in the login keychain."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return _token_from_payload(json.loads(result.stdout))
    except ValueError:
        return None


def _retry_after_epoch(value: Optional[str], now_epoch: int) -> Optional[int]:
    if not value:
        return None
    try:
        seconds = int(value)
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.astimezone(timezone.utc).timestamp())
    return now_epoch + max(0, seconds)


def read_api_limits(
    enabled: bool,
    cache_path: Optional[Path] = None,
    now_epoch: Optional[int] = None,
    ttl_seconds: int = API_TTL_SECONDS,
    opener: Optional[Callable[..., Any]] = None,
    credentials_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Read normalized experimental OAuth usage limits.

    Returns ``None`` unless explicitly enabled. The normalized shape mirrors the
    official statusline reader enough for the snapshot builder to consume it.
    """
    if not enabled:
        return None

    now = int(time.time()) if now_epoch is None else now_epoch
    path = cache_path or default_api_cache_path()
    cache = _read_cache(path)

    retry_after = _finite_int((cache or {}).get("retry_after_epoch"))
    if retry_after is not None and now < retry_after:
        return _cached_limits(cache, now, ttl_seconds)

    if _cache_is_fresh(cache, now, ttl_seconds):
        return _cached_limits(cache, now, ttl_seconds)

    token = read_oauth_token(credentials_path)
    if not token:
        return _cached_limits(cache, now, ttl_seconds)

    user_agent = f"claude-code/{os.environ.get('CLAUDE_CODE_VERSION', 'unknown')}"
    request = urllib.request.Request(
        API_USAGE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": _BETA_HEADER,
            "User-Agent": user_agent,
            "Content-Type": "application/json",
        },
        method="GET",
    )
    call = opener or urllib.request.urlopen

    try:
        with call(request, timeout=10) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            retry_epoch = _retry_after_epoch(exc.headers.get("Retry-After"), now)
            limits = _cached_limits(cache, now, ttl_seconds)
            try:
                _write_cache(path, limits, now, retry_after_epoch=retry_epoch)
            except OSError as write_error:
                logger.debug(
                    "Failed to write experimental API retry cache: %s", write_error
                )
            return limits
        logger.debug("Experimental API request failed with HTTP %s", exc.code)
        return _cached_limits(cache, now, ttl_seconds)
    except (OSError, ValueError) as exc:
        logger.debug("Experimental API request failed: %s", exc)
        return _cached_limits(cache, now, ttl_seconds)

    if not isinstance(payload, dict):
        return _cached_limits(cache, now, ttl_seconds)

    limits = _normalize(payload, captured_at_epoch=now, now_epoch=now)
    try:
        _write_cache(path, limits, now)
    except OSError as exc:
        logger.debug("Failed to write experimental API cache: %s", exc)
    return limits
