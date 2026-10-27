"""Every setting Arrival reads from the environment, in one place (L22).

`get(name)` returns the process environment's value, else one from an untracked `.env` (the cwd first, then the
repo root). A `.env` never overrides a variable that is already set, and ARRIVAL_DOTENV=0 turns the `.env`
lookup off (the test suite sets it, so a developer's real keys can never reach a test). Only `get` returns a
value, for the module that makes the request; `status()` reports set or unset and never the value, and
`redact()` scrubs secret values and common token shapes from text before it is printed, logged or stored.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REDACTED = "[redacted]"


@dataclass(frozen=True)
class Setting:
    used_by: str
    secret: bool = False
    default: str | None = None


SETTINGS: dict[str, Setting] = {
    "TOKEN_FACTORY_API_KEY": Setting("arrival.models.token_factory (Nemotron planner, wordings)", secret=True),
    "TOKEN_FACTORY_BASE_URL": Setting("arrival.models.token_factory", default="https://api.tokenfactory.nebius.com/v1"),
    "NEMOTRON_OMNI_MODEL": Setting("arrival.models.token_factory"),
    "NEMOTRON_ULTRA_MODEL": Setting("arrival.models.token_factory"),
    "NEMOTRON_SUPER_MODEL": Setting("arrival.models.token_factory"),
    "NEMOTRON_LIGHTNING_MODEL": Setting("arrival.models.token_factory"),
    "TAVILY_API_KEY": Setting("arrival.models.tavily (research, sim --tavily)", secret=True),
    "TAVILY_BASE_URL": Setting("arrival.models.tavily", default="https://api.tavily.com/search"),
    "GR00T_POLICY_URL": Setting("arrival.sim.policy.VLAClient (--policy vla)"),
    "COSMOS_URL": Setting("arrival.render.variation.CosmosClient (Cosmos3 Generator endpoint base URL)"),
    "COSMOS_API_KEY": Setting("arrival.render.variation.CosmosClient (hosted endpoint bearer auth)", secret=True),
    "COSMOS_TRANSFER_ENABLED": Setting("arrival.render.variation.CosmosClient (opt in to edge-transfer mode)"),
    "NEBIUS_IAM_TOKEN": Setting("arrival.jobs.nebius (fanout --backend nebius)", secret=True),
    "NEBIUS_API_URL": Setting("arrival.jobs.nebius", default="https://api.nebius.cloud"),
    "NEBIUS_PROJECT_ID": Setting("arrival.jobs.nebius"),
    "NEBIUS_JOB_IMAGE": Setting("arrival.jobs.nebius"),
    "NEBIUS_JOB_PLATFORM": Setting("arrival.jobs.nebius", default="cpu-d3"),
    "NEBIUS_JOB_PRESET": Setting("arrival.jobs.nebius", default="4vcpu-16gb"),
    "NEBIUS_JOB_SUBNET_ID": Setting("arrival.jobs.nebius"),
    "NEBIUS_MAX_JOB_SECONDS": Setting("arrival.jobs.nebius", default="7200"),
    "NEBIUS_MAX_NO_STATE_SECONDS": Setting("arrival.jobs.nebius", default="1800"),
    "NEBIUS_JOB_S3_SECRET_ID": Setting("arrival.jobs.nebius"),
    "NEBIUS_REGION": Setting("arrival.jobs.nebius", default="eu-north1"),
    "NEBIUS_S3_ENDPOINT": Setting("arrival.jobs.nebius"),
    "NEBIUS_BUCKET": Setting("arrival.jobs.nebius"),
    "NEBIUS_ACCESS_KEY_ID": Setting("arrival.jobs.nebius", secret=True),
    "NEBIUS_SECRET_ACCESS_KEY": Setting("arrival.jobs.nebius", secret=True),
}

# token shapes worth scrubbing even when the value is not one of ours (a key pasted into an error message)
_PATTERNS = [
    (re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1" + REDACTED),
    (re.compile(r"\b(nvapi|tvly|sk|hf)-[A-Za-z0-9_-]{8,}"), REDACTED),
    (re.compile(r"(?i)(\"?(?:api[_-]?key|token|secret|password)\"?\s*[:=]\s*\"?)[^\"\s,&}]{4,}"), r"\1" + REDACTED),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1" + REDACTED + "@"),
    (re.compile(r"(?i)([?&](?:token|key|api_key|apikey|access_token|sig|signature)=)[^&#\s]+"), r"\1" + REDACTED),
]


def dotenv_paths() -> list[Path]:
    return [Path.cwd() / ".env", REPO_ROOT / ".env"]


def load_dotenv(paths: list[Path] | None = None) -> None:
    """Tiny stdlib .env reader: KEY=VALUE lines, '#' comments, optional `export` and quotes; variables already
    set win. With no `paths` it reads `dotenv_paths()`, unless ARRIVAL_DOTENV=0."""
    if paths is None:
        if os.environ.get("ARRIVAL_DOTENV", "1") == "0":
            return
        paths = dotenv_paths()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def get(name: str, default: str | None = None) -> str | None:
    """The setting's value: the environment, else `.env`, else `default`, else the setting's own default.
    An empty value counts as unset."""
    if not os.environ.get(name):
        load_dotenv()
    value = os.environ.get(name)
    if value:
        return value
    if default is not None:
        return default
    s = SETTINGS.get(name)
    return s.default if s else None


def status() -> list[dict]:
    """One row per known setting: name, whether it is set, whether it is a secret, who reads it. No values."""
    load_dotenv()
    return [{"name": n, "set": bool(os.environ.get(n)), "secret": s.secret, "used_by": s.used_by,
             "has_default": s.default is not None} for n, s in SETTINGS.items()]


def redact(text) -> str:
    """`text` with the values of every set secret setting and common token shapes replaced by [redacted]."""
    out = str(text)
    for name, s in SETTINGS.items():
        v = os.environ.get(name)
        if s.secret and v and len(v) >= 4:
            out = out.replace(v, REDACTED)
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    return out


class RedactingFilter(logging.Filter):
    """Logging filter that runs every record's final message through `redact`."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = redact(record.getMessage()), ()
        return True


def install_log_redaction(logger: logging.Logger | None = None) -> RedactingFilter:
    """Attach a RedactingFilter to every handler of `logger` (default: the root logger) and return it."""
    logger = logger or logging.getLogger()
    f = RedactingFilter()
    for h in logger.handlers:
        h.addFilter(f)
    logger.addFilter(f)
    return f
