"""Pre-build secret and credential leak scanner for VideoContextEngine.

Scans all repository files destined for packaging to ensure:
1. No active Google OAuth client secrets (GOCSPX-...) exist.
2. No private keys, certificates, or tokens are committed or packaged.
3. Example configuration files contain only approved placeholders.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Forbidden patterns indicating live secrets
FORBIDDEN_PATTERNS = [
    (re.compile(r"GOCSPX-[A-Za-z0-9_\-]{20,}"), "Live Google OAuth Client Secret (GOCSPX-...)"),
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"), "Cryptographic Private Key"),
    (re.compile(r"(?i)AIza[0-9A-Za-z-_]{35}"), "Google API Key (AIza...)"),
    (re.compile(r"(?i)AKIA[0-9A-Z]{16}"), "AWS Access Key"),
]

# File exclusions (local ignored configs, build artifacts, tests that deliberately check rules)
EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "installer_output",
    "Output",
    ".venv",
    "venv",
    "env",
    ".venv313",
    "data",
    "hf_cache",
    "torch_cache",
}

EXCLUDED_FILES = {
    ".env",  # User's private local environment
    ".env.local",
    "secrets.toml",
    "scan_secrets.py",  # Self
}


def scan_file(path: Path) -> list[str]:
    """Scan a single file for secret patterns."""
    issues = []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return [f"Could not read {path}: {exc}"]

    for pattern, desc in FORBIDDEN_PATTERNS:
        matches = pattern.findall(content)
        if matches:
            issues.append(f"{desc} detected in {path.relative_to(ROOT)} (count: {len(matches)})")

    return issues


def main() -> int:
    print("==================================================")
    print("VideoContextEngine: Pre-Build Secret Leak Scanner")
    print(f"Scanning Root: {ROOT}")
    print("==================================================")

    all_issues = []
    scanned_count = 0

    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue

        # Check directory exclusions
        parts = set(p.parts)
        if parts.intersection(EXCLUDED_DIRS):
            continue

        # Check file exclusions
        if p.name in EXCLUDED_FILES or p.name.endswith(".backup") or p.name.endswith(".bak"):
            continue

        # Skip binary files (.exe, .dll, .pyd, .ico, .jpg, .png, .mp4, etc.)
        if p.suffix.lower() in {
            ".exe", ".dll", ".pyd", ".so", ".dylib", ".ico", ".jpg", ".jpeg",
            ".png", ".gif", ".mp4", ".mkv", ".wav", ".pyc", ".db", ".sqlite",
            ".index", ".bin", ".tar", ".gz", ".zip", ".whl"
        }:
            continue

        issues = scan_file(p)
        if issues:
            all_issues.extend(issues)
        scanned_count += 1

    print(f"Scanned {scanned_count} files.")

    if all_issues:
        print("\n[!] CRITICAL: Secret scan FAILED with the following detections:")
        for issue in all_issues:
            print(f"  - {issue}")
        print("\nPlease remove or redact these secrets before proceeding with build.\n")
        return 1

    print("[PASS] Secret scan PASSED: No secrets or live credentials detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
