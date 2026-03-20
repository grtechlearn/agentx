#!/usr/bin/env python3
"""
AgentX Secret Scanner — Pre-commit hook to prevent accidental secret exposure.

Scans staged files for API keys, tokens, passwords, and other secrets.
Run manually: python scripts/check_secrets.py
Install as pre-commit: cp scripts/check_secrets.py .git/hooks/pre-commit

This uses the same patterns as AgentX's VulnerabilityScanner.
"""

import re
import subprocess
import sys

# Patterns that should NEVER appear in committed code
SECRET_PATTERNS = [
    # API Keys
    (r"sk-[a-zA-Z0-9]{32,}", "OpenAI/Anthropic API key"),
    (r"sk-proj-[a-zA-Z0-9]{32,}", "OpenAI project API key"),
    (r"sk-ant-[a-zA-Z0-9]{32,}", "Anthropic API key"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"AIza[a-zA-Z0-9_-]{35}", "Google API key"),
    (r"xox[bprs]-[a-zA-Z0-9-]+", "Slack token"),

    # Private keys
    (r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----", "Private key"),

    # Connection strings with passwords
    (r"(?:postgres|mysql|mongodb)://[^:]+:[^@]+@[^/]+", "Database connection string with password"),

    # Specific server IPs (your Hetzner VPS)
    (r"89\.167\.79\.10", "Hetzner server IP"),
]

# Files/patterns to skip
SKIP_PATTERNS = [
    r"scripts/check_secrets\.py",  # This file itself
    r"agentx/security/moderation\.py",  # Contains regex patterns (not actual secrets)
    r"\.pyc$",
    r"__pycache__",
    r"\.git/",
    r"dist/",
    r"\.egg-info",
]


def get_staged_files() -> list[str]:
    """Get list of staged files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True, text=True
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        return []


def should_skip(filepath: str) -> bool:
    """Check if file should be skipped."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, filepath):
            return True
    return False


def scan_file(filepath: str) -> list[tuple[int, str, str]]:
    """Scan a file for secrets. Returns [(line_num, pattern_desc, matched_text)]."""
    findings = []
    try:
        with open(filepath, "r", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                # Skip comments and string constructions
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue
                # Skip lines that dynamically construct test values
                if "fake_" in line or "construct" in line or "dynamically" in line:
                    continue
                if 'f"' in line and "{fake_" in line:
                    continue

                for pattern, desc in SECRET_PATTERNS:
                    if re.search(pattern, line):
                        # Extract the match for display (truncated)
                        match = re.search(pattern, line)
                        matched = match.group(0) if match else ""
                        # Redact most of it
                        if len(matched) > 8:
                            matched = matched[:4] + "..." + matched[-4:]
                        findings.append((line_num, desc, matched))
    except (OSError, UnicodeDecodeError):
        pass
    return findings


def main() -> int:
    """Run the secret scanner. Returns 0 if clean, 1 if secrets found."""
    # If run as pre-commit hook, scan staged files
    staged = get_staged_files()

    # If no staged files (manual run), scan all Python files
    if not staged:
        import glob
        staged = glob.glob("**/*.py", recursive=True)
        staged += glob.glob("**/*.yml", recursive=True)
        staged += glob.glob("**/*.yaml", recursive=True)
        staged += glob.glob("**/*.json", recursive=True)
        staged += glob.glob("**/*.md", recursive=True)
        staged += glob.glob("**/*.html", recursive=True)

    total_findings = 0
    for filepath in staged:
        if should_skip(filepath):
            continue

        findings = scan_file(filepath)
        if findings:
            total_findings += len(findings)
            print(f"\n{'='*60}")
            print(f"SECRETS DETECTED in {filepath}")
            print(f"{'='*60}")
            for line_num, desc, matched in findings:
                print(f"  Line {line_num}: {desc} ({matched})")

    if total_findings > 0:
        print(f"\n{'!'*60}")
        print(f"  BLOCKED: {total_findings} potential secret(s) found!")
        print(f"  Remove secrets before committing.")
        print(f"  Use environment variables instead.")
        print(f"{'!'*60}\n")
        return 1

    print("Secret scan: CLEAN (no secrets detected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
