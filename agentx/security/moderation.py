"""
AgentX - Content Moderation & Vulnerability Scanner.
Configurable content safety layer for filtering abuse, profanity, sexual content,
and detecting security vulnerabilities in user input/output.

All word lists and categories are fully configurable — add custom lists,
enable/disable categories, set severity levels, choose actions (block/warn/redact).

Usage:
    # Default configuration
    moderator = ContentModerator()
    result = moderator.check("some user input")
    if result.blocked:
        print(f"Blocked: {result.reason}")

    # Custom configuration
    moderator = ContentModerator(ModerationConfig(
        categories={
            "profanity": CategoryConfig(enabled=True, action="block"),
            "sexual": CategoryConfig(enabled=True, action="redact"),
            "abuse": CategoryConfig(enabled=True, action="warn"),
            "violence": CategoryConfig(enabled=False),
        },
        custom_blocked_words=["custom_bad_word"],
        severity_threshold="medium",
    ))

    # Vulnerability scanning
    scanner = VulnerabilityScanner()
    result = scanner.scan("user input with <script>alert(1)</script>")
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModerationAction(str, Enum):
    """Action to take when moderation triggers."""
    BLOCK = "block"       # Reject entirely
    WARN = "warn"         # Allow but flag
    REDACT = "redact"     # Replace matched content with ***
    LOG = "log"           # Silent log only


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------

class CategoryConfig(BaseModel):
    """Configuration for a single moderation category."""
    enabled: bool = True
    action: ModerationAction = ModerationAction.BLOCK
    severity: Severity = Severity.MEDIUM
    custom_words: list[str] = Field(default_factory=list)
    custom_patterns: list[str] = Field(default_factory=list)


class ModerationConfig(BaseModel):
    """
    Full moderation configuration — pluggable like LLM models.

    Categories:
    - profanity: Common profanity and swear words
    - sexual: Sexual/explicit content
    - abuse: Hate speech, slurs, harassment
    - violence: Violent threats and graphic content
    - self_harm: Self-harm and suicide content
    - drugs: Drug-related content
    - custom: User-defined blocked content
    """
    enabled: bool = True
    severity_threshold: Severity = Severity.LOW  # block at this severity and above
    default_action: ModerationAction = ModerationAction.BLOCK

    # Per-category configuration
    categories: dict[str, CategoryConfig] = Field(default_factory=lambda: {
        "profanity": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.MEDIUM),
        "sexual": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.HIGH),
        "abuse": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.HIGH),
        "violence": CategoryConfig(enabled=True, action=ModerationAction.WARN, severity=Severity.MEDIUM),
        "self_harm": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.CRITICAL),
        "drugs": CategoryConfig(enabled=False, action=ModerationAction.WARN, severity=Severity.LOW),
        "custom": CategoryConfig(enabled=False, action=ModerationAction.BLOCK, severity=Severity.MEDIUM),
    })

    # Global custom blocked words (added to all categories)
    custom_blocked_words: list[str] = Field(default_factory=list)
    custom_blocked_patterns: list[str] = Field(default_factory=list)

    # Whitelist — these words/phrases are never flagged
    whitelist_words: list[str] = Field(default_factory=list)
    whitelist_patterns: list[str] = Field(default_factory=list)

    # Behavior
    check_input: bool = True    # Check user input
    check_output: bool = True   # Check LLM output
    case_sensitive: bool = False
    log_violations: bool = True
    max_violations_before_ban: int = 0  # 0 = no auto-ban

    @classmethod
    def strict(cls) -> ModerationConfig:
        """Strict moderation — all categories enabled, block everything."""
        return cls(
            enabled=True,
            severity_threshold=Severity.LOW,
            default_action=ModerationAction.BLOCK,
            categories={
                "profanity": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.MEDIUM),
                "sexual": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.HIGH),
                "abuse": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.HIGH),
                "violence": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.MEDIUM),
                "self_harm": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.CRITICAL),
                "drugs": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.MEDIUM),
                "custom": CategoryConfig(enabled=False),
            },
        )

    @classmethod
    def moderate(cls) -> ModerationConfig:
        """Moderate — block sexual/abuse, warn on profanity/violence."""
        return cls(
            enabled=True,
            severity_threshold=Severity.MEDIUM,
            default_action=ModerationAction.WARN,
            categories={
                "profanity": CategoryConfig(enabled=True, action=ModerationAction.WARN, severity=Severity.LOW),
                "sexual": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.HIGH),
                "abuse": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.HIGH),
                "violence": CategoryConfig(enabled=True, action=ModerationAction.WARN, severity=Severity.MEDIUM),
                "self_harm": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.CRITICAL),
                "drugs": CategoryConfig(enabled=False),
                "custom": CategoryConfig(enabled=False),
            },
        )

    @classmethod
    def permissive(cls) -> ModerationConfig:
        """Permissive — only block critical content (sexual, self-harm)."""
        return cls(
            enabled=True,
            severity_threshold=Severity.HIGH,
            default_action=ModerationAction.LOG,
            categories={
                "profanity": CategoryConfig(enabled=False),
                "sexual": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.HIGH),
                "abuse": CategoryConfig(enabled=True, action=ModerationAction.WARN, severity=Severity.HIGH),
                "violence": CategoryConfig(enabled=False),
                "self_harm": CategoryConfig(enabled=True, action=ModerationAction.BLOCK, severity=Severity.CRITICAL),
                "drugs": CategoryConfig(enabled=False),
                "custom": CategoryConfig(enabled=False),
            },
        )

    @classmethod
    def disabled(cls) -> ModerationConfig:
        """Disabled — no moderation."""
        return cls(enabled=False)


# ---------------------------------------------------------------------------
# Moderation Result
# ---------------------------------------------------------------------------

class ModerationResult(BaseModel):
    """Result of content moderation check."""
    blocked: bool = False
    action: ModerationAction = ModerationAction.LOG
    reason: str = ""
    categories_triggered: list[str] = Field(default_factory=list)
    matched_words: list[str] = Field(default_factory=list)
    severity: Severity = Severity.LOW
    sanitized_text: str = ""
    details: list[dict[str, Any]] = Field(default_factory=list)


class VulnerabilityResult(BaseModel):
    """Result of vulnerability scanning."""
    has_vulnerabilities: bool = False
    blocked: bool = False
    vulnerabilities: list[dict[str, Any]] = Field(default_factory=list)
    severity: Severity = Severity.LOW
    sanitized_text: str = ""


# ---------------------------------------------------------------------------
# Built-in Word Lists (minimal seed — users add their own)
# ---------------------------------------------------------------------------

# These are minimal representative samples. In production, load from file/DB.
# Words are stored as lowercase stems for matching.

_PROFANITY_WORDS: set[str] = {
    "fuck", "shit", "damn", "ass", "bitch", "bastard", "crap",
    "piss", "dick", "cock", "pussy", "motherfucker", "fucker",
    "asshole", "bullshit", "horseshit", "dumbass", "jackass",
    "wtf", "stfu", "lmfao",
}

_SEXUAL_WORDS: set[str] = {
    "porn", "pornography", "hentai", "xxx", "nsfw", "nude", "nudes",
    "naked", "sex", "sexual", "explicit", "erotic", "orgasm",
    "masturbat", "genitals", "fetish", "bondage", "incest",
    "pedophil", "rape", "molest",
}

_ABUSE_WORDS: set[str] = {
    "nigger", "nigga", "faggot", "fag", "retard", "retarded",
    "kike", "spic", "chink", "gook", "wetback", "raghead",
    "tranny", "dyke", "cunt", "whore", "slut",
    "kill yourself", "kys", "die",
}

_VIOLENCE_WORDS: set[str] = {
    "murder", "kill", "stab", "shoot", "bomb", "terrorist",
    "massacre", "slaughter", "decapitat", "torture", "mutilat",
    "bloodbath", "genocide", "assassinat",
}

_SELF_HARM_WORDS: set[str] = {
    "suicide", "suicidal", "self-harm", "self harm", "cut myself",
    "kill myself", "end my life", "want to die", "overdose",
    "hang myself", "slit wrist",
}

_DRUGS_WORDS: set[str] = {
    "cocaine", "heroin", "meth", "methamphetamine", "crack",
    "lsd", "ecstasy", "mdma", "fentanyl", "opioid",
    "drug deal", "drug lord",
}

_CATEGORY_WORD_LISTS: dict[str, set[str]] = {
    "profanity": _PROFANITY_WORDS,
    "sexual": _SEXUAL_WORDS,
    "abuse": _ABUSE_WORDS,
    "violence": _VIOLENCE_WORDS,
    "self_harm": _SELF_HARM_WORDS,
    "drugs": _DRUGS_WORDS,
}

# Patterns for more nuanced matching (regex)
_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "profanity": [
        r"\bf+u+c+k+\b",
        r"\bs+h+i+t+\b",
        r"\ba+s+s+\b(?!ess|ist|ign|ume|ert|oci|embl)",
    ],
    "sexual": [
        r"(?i)\bsex(?:ual|ting|ts)?\b(?!\s+(?:ism|ist|ual\s+(?:harassment|assault|orientation)))",
        r"(?i)\bnud(?:e|es|ity)\b",
        r"(?i)\bporn(?:ograph(?:y|ic))?\b",
    ],
    "abuse": [
        r"(?i)\b(?:hate|hating)\s+(?:all\s+)?(?:women|men|blacks|whites|jews|muslims|gays|trans)\b",
        r"(?i)\b(?:go\s+)?(?:die|kill\s+yourself|kys)\b",
    ],
    "violence": [
        r"(?i)\b(?:i'?ll?|gonna|going\s+to)\s+(?:kill|murder|shoot|stab|bomb)\b",
        r"(?i)\b(?:threat(?:en)?(?:ing)?)\s+(?:to\s+)?(?:kill|harm|hurt)\b",
    ],
    "self_harm": [
        r"(?i)\b(?:want|going|plan(?:ning)?)\s+to\s+(?:die|kill\s+myself|end\s+(?:it|my\s+life))\b",
        r"(?i)\b(?:cut(?:ting)?|harm(?:ing)?)\s+(?:myself|my\s+(?:wrist|arm))\b",
    ],
}


# ---------------------------------------------------------------------------
# Content Moderator
# ---------------------------------------------------------------------------

class ContentModerator:
    """
    Configurable content moderation engine.

    Checks text against word lists and patterns across categories.
    Fully configurable — enable/disable categories, add custom words,
    set per-category actions (block/warn/redact/log).

    Like LLM models, moderation presets are pluggable:
        ContentModerator(ModerationConfig.strict())
        ContentModerator(ModerationConfig.moderate())
        ContentModerator(ModerationConfig.permissive())
        ContentModerator(ModerationConfig.disabled())
    """

    def __init__(self, config: ModerationConfig | None = None):
        self.config = config or ModerationConfig()
        self._whitelist_words: set[str] = set(w.lower() for w in self.config.whitelist_words)
        self._whitelist_patterns = [re.compile(p, re.IGNORECASE) for p in self.config.whitelist_patterns]
        self._custom_patterns = [re.compile(p, re.IGNORECASE) for p in self.config.custom_blocked_patterns]
        self._custom_words: set[str] = set(w.lower() for w in self.config.custom_blocked_words)
        self._stats: dict[str, int] = {"total_checks": 0, "total_blocked": 0, "total_warned": 0, "by_category": {}}
        self._user_violations: dict[str, int] = {}

        # Pre-compile category patterns
        self._compiled_patterns: dict[str, list[re.Pattern[str]]] = {}
        for cat, patterns in _CATEGORY_PATTERNS.items():
            cat_cfg = self.config.categories.get(cat)
            if cat_cfg and cat_cfg.enabled:
                compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
                # Add custom patterns from category config
                for cp in cat_cfg.custom_patterns:
                    compiled.append(re.compile(cp, re.IGNORECASE))
                self._compiled_patterns[cat] = compiled

    def check(self, text: str, user_id: str = "") -> ModerationResult:
        """
        Check text for content violations.

        Returns ModerationResult with blocking decision, matched categories,
        and sanitized text if action is REDACT.
        """
        if not self.config.enabled:
            return ModerationResult(sanitized_text=text)

        self._stats["total_checks"] += 1
        result = ModerationResult(sanitized_text=text)
        check_text = text if self.config.case_sensitive else text.lower()

        # Check whitelist — whitelisted content passes through
        if self._is_whitelisted(text):
            return result

        triggered: list[dict[str, Any]] = []

        # Check each enabled category
        for cat_name, cat_cfg in self.config.categories.items():
            if not cat_cfg.enabled:
                continue

            matches = self._check_category(check_text, cat_name, cat_cfg)
            if matches:
                triggered.append({
                    "category": cat_name,
                    "action": cat_cfg.action.value,
                    "severity": cat_cfg.severity.value,
                    "matches": matches,
                })

        # Check global custom words
        custom_matches = self._check_custom_words(check_text)
        if custom_matches:
            triggered.append({
                "category": "custom",
                "action": self.config.default_action.value,
                "severity": Severity.MEDIUM.value,
                "matches": custom_matches,
            })

        # Check global custom patterns
        pattern_matches = self._check_custom_patterns(text)
        if pattern_matches:
            triggered.append({
                "category": "custom_pattern",
                "action": self.config.default_action.value,
                "severity": Severity.MEDIUM.value,
                "matches": pattern_matches,
            })

        if not triggered:
            return result

        # Determine overall action (most severe wins)
        action_priority = {
            ModerationAction.LOG: 0,
            ModerationAction.WARN: 1,
            ModerationAction.REDACT: 2,
            ModerationAction.BLOCK: 3,
        }
        severity_priority = {
            Severity.LOW: 0,
            Severity.MEDIUM: 1,
            Severity.HIGH: 2,
            Severity.CRITICAL: 3,
        }

        max_action = ModerationAction.LOG
        max_severity = Severity.LOW
        all_matches: list[str] = []
        all_categories: list[str] = []

        for t in triggered:
            action = ModerationAction(t["action"])
            severity = Severity(t["severity"])
            if action_priority[action] > action_priority[max_action]:
                max_action = action
            if severity_priority[severity] > severity_priority[max_severity]:
                max_severity = severity
            all_matches.extend(t["matches"])
            all_categories.append(t["category"])

        result.action = max_action
        result.severity = max_severity
        result.categories_triggered = list(set(all_categories))
        result.matched_words = list(set(all_matches))
        result.details = triggered
        result.reason = f"Content violation: {', '.join(result.categories_triggered)}"
        result.blocked = max_action == ModerationAction.BLOCK

        # Apply redaction if needed
        if max_action == ModerationAction.REDACT:
            result.sanitized_text = self._redact(text, all_matches)
        elif max_action == ModerationAction.BLOCK:
            result.sanitized_text = "[CONTENT BLOCKED]"

        # Stats
        if result.blocked:
            self._stats["total_blocked"] += 1
        if max_action == ModerationAction.WARN:
            self._stats["total_warned"] += 1

        for cat in all_categories:
            self._stats["by_category"][cat] = self._stats["by_category"].get(cat, 0) + 1

        # Track user violations
        if user_id and result.blocked:
            self._user_violations[user_id] = self._user_violations.get(user_id, 0) + 1

        if self.config.log_violations:
            logger.warning(
                f"Content moderation: action={max_action.value} categories={all_categories} "
                f"severity={max_severity.value} user={user_id or 'unknown'}"
            )

        return result

    def is_user_banned(self, user_id: str) -> bool:
        """Check if user has exceeded violation threshold."""
        if self.config.max_violations_before_ban <= 0:
            return False
        return self._user_violations.get(user_id, 0) >= self.config.max_violations_before_ban

    def add_words(self, category: str, words: list[str]) -> None:
        """Dynamically add words to a category."""
        if category in _CATEGORY_WORD_LISTS:
            _CATEGORY_WORD_LISTS[category].update(w.lower() for w in words)
        elif category == "custom":
            self._custom_words.update(w.lower() for w in words)

    def remove_words(self, category: str, words: list[str]) -> None:
        """Remove words from a category."""
        if category in _CATEGORY_WORD_LISTS:
            _CATEGORY_WORD_LISTS[category] -= set(w.lower() for w in words)
        elif category == "custom":
            self._custom_words -= set(w.lower() for w in words)

    def add_whitelist(self, words: list[str]) -> None:
        """Add words to whitelist."""
        self._whitelist_words.update(w.lower() for w in words)

    def add_pattern(self, category: str, pattern: str) -> None:
        """Add a regex pattern to a category."""
        compiled = re.compile(pattern, re.IGNORECASE)
        if category not in self._compiled_patterns:
            self._compiled_patterns[category] = []
        self._compiled_patterns[category].append(compiled)

    def stats(self) -> dict[str, Any]:
        """Get moderation statistics."""
        return {
            **self._stats,
            "block_rate": (
                self._stats["total_blocked"] / self._stats["total_checks"]
                if self._stats["total_checks"] > 0 else 0.0
            ),
            "users_with_violations": len(self._user_violations),
        }

    # --- Internal ---

    def _is_whitelisted(self, text: str) -> bool:
        """Check if text is whitelisted."""
        lower = text.lower()
        for word in self._whitelist_words:
            if word in lower:
                return True
        for pattern in self._whitelist_patterns:
            if pattern.search(text):
                return True
        return False

    def _check_category(self, text: str, cat_name: str, cat_cfg: CategoryConfig) -> list[str]:
        """Check text against a category's word list and patterns."""
        matches: list[str] = []

        # Word list matching
        word_list = _CATEGORY_WORD_LISTS.get(cat_name, set())
        # Merge per-category custom words
        all_words = word_list | set(w.lower() for w in cat_cfg.custom_words)

        for word in all_words:
            if word in text:
                matches.append(word)

        # Pattern matching
        patterns = self._compiled_patterns.get(cat_name, [])
        for pattern in patterns:
            m = pattern.search(text)
            if m:
                matches.append(m.group())

        return matches

    def _check_custom_words(self, text: str) -> list[str]:
        """Check against global custom blocked words."""
        matches = []
        for word in self._custom_words:
            if word in text:
                matches.append(word)
        return matches

    def _check_custom_patterns(self, text: str) -> list[str]:
        """Check against global custom patterns."""
        matches = []
        for pattern in self._custom_patterns:
            m = pattern.search(text)
            if m:
                matches.append(m.group())
        return matches

    def _redact(self, text: str, matches: list[str]) -> str:
        """Replace matched words with *** in text."""
        redacted = text
        for word in sorted(matches, key=len, reverse=True):
            redacted = re.sub(re.escape(word), "***", redacted, flags=re.IGNORECASE)
        return redacted


# ---------------------------------------------------------------------------
# Vulnerability Scanner
# ---------------------------------------------------------------------------

class VulnerabilityScanner:
    """
    Scans content for security vulnerabilities.

    Detects:
    1. Code injection (eval, exec, os.system patterns)
    2. Credential exposure (API keys, passwords, tokens in text)
    3. Unsafe URLs (data:, javascript:, file://)
    4. Serialization attacks (pickle, yaml.load)
    5. Information disclosure (internal IPs, stack traces, debug info)
    6. Insecure patterns (hardcoded secrets, weak crypto)

    Configurable — enable/disable categories, set custom patterns.
    """

    # Code injection patterns
    CODE_INJECTION_PATTERNS: list[tuple[str, str, Severity]] = [
        (r"(?i)\beval\s*\(", "eval() call", Severity.HIGH),
        (r"(?i)\bexec\s*\(", "exec() call", Severity.HIGH),
        (r"(?i)\bos\.system\s*\(", "os.system() call", Severity.CRITICAL),
        (r"(?i)\bsubprocess\s*\.\s*(?:call|run|Popen)\s*\(", "subprocess call", Severity.CRITICAL),
        (r"(?i)\b__import__\s*\(", "__import__() call", Severity.HIGH),
        (r"(?i)\bcompile\s*\(.*\bexec\b", "compile+exec", Severity.HIGH),
        (r"(?i)\bglobals\s*\(\s*\)\s*\[", "globals() access", Severity.MEDIUM),
        (r"(?i)\bsetattr\s*\(", "setattr() call", Severity.MEDIUM),
    ]

    # Credential exposure patterns
    CREDENTIAL_PATTERNS: list[tuple[str, str, Severity]] = [
        (r"(?i)(?:api[_-]?key|apikey)\s*[=:]\s*['\"][a-zA-Z0-9_\-]{20,}", "API key exposed", Severity.CRITICAL),
        (r"(?i)(?:password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{4,}", "Password exposed", Severity.CRITICAL),
        (r"(?i)(?:secret|token)\s*[=:]\s*['\"][a-zA-Z0-9_\-]{16,}", "Secret/token exposed", Severity.CRITICAL),
        (r"(?:sk-[a-zA-Z0-9]{32,})", "OpenAI API key", Severity.CRITICAL),
        (r"(?:ghp_[a-zA-Z0-9]{36})", "GitHub personal access token", Severity.CRITICAL),
        (r"(?:AKIA[0-9A-Z]{16})", "AWS access key", Severity.CRITICAL),
        (r"(?i)(?:bearer|authorization)\s+[a-zA-Z0-9_\-\.]{20,}", "Auth token exposed", Severity.HIGH),
        (r"(?:-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----)", "Private key exposed", Severity.CRITICAL),
    ]

    # Unsafe URL patterns
    UNSAFE_URL_PATTERNS: list[tuple[str, str, Severity]] = [
        (r"(?i)javascript\s*:", "javascript: URL", Severity.HIGH),
        (r"(?i)data\s*:\s*text/html", "data: HTML URL", Severity.HIGH),
        (r"(?i)file\s*:///", "file:// URL", Severity.MEDIUM),
        (r"(?i)vbscript\s*:", "vbscript: URL", Severity.HIGH),
    ]

    # Serialization attack patterns
    SERIALIZATION_PATTERNS: list[tuple[str, str, Severity]] = [
        (r"(?i)\bpickle\.loads?\s*\(", "Unsafe pickle deserialization", Severity.HIGH),
        (r"(?i)\byaml\.(?:load|unsafe_load)\s*\(", "Unsafe YAML load", Severity.HIGH),
        (r"(?i)\bmarshal\.loads?\s*\(", "Unsafe marshal load", Severity.HIGH),
        (r"(?i)\bjsonpickle\.", "jsonpickle usage", Severity.MEDIUM),
    ]

    # Information disclosure patterns
    INFO_DISCLOSURE_PATTERNS: list[tuple[str, str, Severity]] = [
        (r"\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b", "Internal IP address", Severity.MEDIUM),
        (r"(?i)(?:traceback|stack\s*trace|exception\s+in\s+thread)", "Stack trace leaked", Severity.MEDIUM),
        (r"(?i)(?:debug\s*=\s*true|debug\s+mode\s+(?:on|enabled))", "Debug mode enabled", Severity.LOW),
        (r"(?i)(?:root|admin):[^:]*:[0-9]+:[0-9]+:", "System user info leaked", Severity.HIGH),
    ]

    # Insecure patterns
    INSECURE_PATTERNS: list[tuple[str, str, Severity]] = [
        (r"(?i)\bmd5\s*\(", "Weak hash (MD5)", Severity.MEDIUM),
        (r"(?i)\bsha1\s*\(", "Weak hash (SHA1)", Severity.LOW),
        (r"(?i)\bDES\b.*(?:encrypt|cipher)", "Weak encryption (DES)", Severity.MEDIUM),
        (r"(?i)(?:verify\s*=\s*False|ssl\s*=\s*False)", "SSL verification disabled", Severity.HIGH),
        (r"(?i)\ballow_redirects\s*=\s*True\b.*\bverify\s*=\s*False", "Unsafe redirect + no SSL", Severity.HIGH),
    ]

    def __init__(
        self,
        check_code_injection: bool = True,
        check_credentials: bool = True,
        check_unsafe_urls: bool = True,
        check_serialization: bool = True,
        check_info_disclosure: bool = True,
        check_insecure_patterns: bool = True,
        custom_patterns: list[tuple[str, str, str]] | None = None,
        block_on_critical: bool = True,
        block_on_high: bool = False,
    ):
        self.check_code_injection = check_code_injection
        self.check_credentials = check_credentials
        self.check_unsafe_urls = check_unsafe_urls
        self.check_serialization = check_serialization
        self.check_info_disclosure = check_info_disclosure
        self.check_insecure_patterns = check_insecure_patterns
        self.block_on_critical = block_on_critical
        self.block_on_high = block_on_high

        # Custom patterns: (regex, description, severity)
        self._custom_patterns: list[tuple[re.Pattern[str], str, Severity]] = []
        for pattern, desc, sev in (custom_patterns or []):
            self._custom_patterns.append((re.compile(pattern, re.IGNORECASE), desc, Severity(sev)))

        self._stats: dict[str, int] = {"total_scans": 0, "total_vulnerabilities": 0, "by_type": {}}

    def scan(self, text: str) -> VulnerabilityResult:
        """
        Scan text for security vulnerabilities.
        Returns VulnerabilityResult with findings.
        """
        self._stats["total_scans"] += 1
        result = VulnerabilityResult(sanitized_text=text)
        findings: list[dict[str, Any]] = []

        # Run each enabled check
        checks: list[tuple[bool, list[tuple[str, str, Severity]]]] = [
            (self.check_code_injection, self.CODE_INJECTION_PATTERNS),
            (self.check_credentials, self.CREDENTIAL_PATTERNS),
            (self.check_unsafe_urls, self.UNSAFE_URL_PATTERNS),
            (self.check_serialization, self.SERIALIZATION_PATTERNS),
            (self.check_info_disclosure, self.INFO_DISCLOSURE_PATTERNS),
            (self.check_insecure_patterns, self.INSECURE_PATTERNS),
        ]

        for enabled, patterns in checks:
            if not enabled:
                continue
            for pattern, desc, severity in patterns:
                m = re.search(pattern, text)
                if m:
                    findings.append({
                        "description": desc,
                        "severity": severity.value,
                        "match": m.group(),
                        "position": m.start(),
                    })

        # Custom patterns
        for pattern, desc, severity in self._custom_patterns:
            m = pattern.search(text)
            if m:
                findings.append({
                    "description": desc,
                    "severity": severity.value,
                    "match": m.group(),
                    "position": m.start(),
                })

        if not findings:
            return result

        result.has_vulnerabilities = True
        result.vulnerabilities = findings
        self._stats["total_vulnerabilities"] += len(findings)

        # Determine max severity
        severity_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        max_sev = Severity.LOW
        for f in findings:
            sev = Severity(f["severity"])
            if severity_order.index(sev) > severity_order.index(max_sev):
                max_sev = sev

        result.severity = max_sev

        # Determine blocking
        if self.block_on_critical and max_sev == Severity.CRITICAL:
            result.blocked = True
        elif self.block_on_high and max_sev in (Severity.HIGH, Severity.CRITICAL):
            result.blocked = True

        # Redact credentials in sanitized text
        if self.check_credentials:
            sanitized = text
            for pattern, desc, _ in self.CREDENTIAL_PATTERNS:
                sanitized = re.sub(pattern, "[REDACTED]", sanitized)
            result.sanitized_text = sanitized

        # Stats
        for f in findings:
            key = f["description"]
            self._stats["by_type"][key] = self._stats["by_type"].get(key, 0) + 1

        logger.warning(
            f"Vulnerability scan: {len(findings)} issues found, "
            f"max_severity={max_sev.value}, blocked={result.blocked}"
        )

        return result

    def add_pattern(self, pattern: str, description: str, severity: str = "medium") -> None:
        """Add a custom vulnerability pattern."""
        self._custom_patterns.append(
            (re.compile(pattern, re.IGNORECASE), description, Severity(severity))
        )

    def stats(self) -> dict[str, Any]:
        """Get vulnerability scanning statistics."""
        return self._stats.copy()
