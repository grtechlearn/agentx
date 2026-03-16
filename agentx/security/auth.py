"""
AgentX - JWT Authentication & Injection Guard.
Secure gateway for all agent requests.

Features:
- JWT token generation & validation
- Token refresh & expiry
- Injection guard (prompt injection, SQL injection, XSS)
- Request signing
- Namespace-scoped access (RBAC decides which vector namespaces user may search)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
import base64
from typing import Any
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger("agentx")


# --- JWT Authentication ---

class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    API = "api"


class JWTToken(BaseModel):
    """JWT-like token payload."""

    sub: str  # user_id
    role: str = "user"
    permissions: list[str] = Field(default_factory=list)
    namespaces: list[str] = Field(default_factory=list)  # allowed vector namespaces
    exp: float = 0.0  # expiry timestamp
    iat: float = Field(default_factory=time.time)  # issued at
    token_type: str = "access"
    jti: str = ""  # token ID for revocation
    org_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuthGateway:
    """
    JWT-based authentication gateway for AgentX.

    Every query passes through this gate:
    1. Validate JWT token
    2. Check injection attacks
    3. RBAC decides allowed namespaces
    4. PII scrubbing (optional)

    Usage:
        auth = AuthGateway(secret_key="your-secret")
        token = auth.create_token(user_id="user-1", role="user")
        result = auth.authenticate(token)
        if result.authenticated:
            # proceed with result.user_id, result.namespaces
    """

    def __init__(
        self,
        secret_key: str = "agentx-jwt-secret",
        access_token_ttl: int = 3600,       # 1 hour
        refresh_token_ttl: int = 86400 * 7,  # 7 days
        algorithm: str = "HS256",
    ):
        self._secret = secret_key.encode()
        self._access_ttl = access_token_ttl
        self._refresh_ttl = refresh_token_ttl
        self._algorithm = algorithm
        self._revoked_tokens: set[str] = set()
        self._injection_guard = InjectionGuard()

    # --- Token Management ---

    def create_token(
        self,
        user_id: str,
        role: str = "user",
        permissions: list[str] | None = None,
        namespaces: list[str] | None = None,
        token_type: str = "access",
        org_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a signed JWT token."""
        ttl = self._access_ttl if token_type == "access" else self._refresh_ttl
        now = time.time()

        payload = JWTToken(
            sub=user_id,
            role=role,
            permissions=permissions or [],
            namespaces=namespaces or ["default"],
            exp=now + ttl,
            iat=now,
            token_type=token_type,
            jti=hashlib.sha256(f"{user_id}:{now}:{token_type}".encode()).hexdigest()[:16],
            org_id=org_id,
            metadata=metadata or {},
        )

        return self._encode(payload.model_dump())

    def create_token_pair(
        self,
        user_id: str,
        role: str = "user",
        permissions: list[str] | None = None,
        namespaces: list[str] | None = None,
        org_id: str = "",
    ) -> dict[str, str]:
        """Create access + refresh token pair."""
        return {
            "access_token": self.create_token(
                user_id, role, permissions, namespaces, "access", org_id,
            ),
            "refresh_token": self.create_token(
                user_id, role, permissions, namespaces, "refresh", org_id,
            ),
        }

    def validate_token(self, token: str) -> JWTToken | None:
        """Validate and decode a JWT token. Returns None if invalid."""
        payload = self._decode(token)
        if payload is None:
            return None

        jwt_token = JWTToken(**payload)

        # Check expiry
        if jwt_token.exp < time.time():
            logger.warning(f"Token expired for user {jwt_token.sub}")
            return None

        # Check revocation
        if jwt_token.jti in self._revoked_tokens:
            logger.warning(f"Revoked token used by {jwt_token.sub}")
            return None

        return jwt_token

    def refresh_token(self, refresh_token_str: str) -> str | None:
        """Use a refresh token to get a new access token."""
        token = self.validate_token(refresh_token_str)
        if token is None or token.token_type != "refresh":
            return None

        return self.create_token(
            user_id=token.sub,
            role=token.role,
            permissions=token.permissions,
            namespaces=token.namespaces,
            token_type="access",
            org_id=token.org_id,
        )

    def revoke_token(self, token_str: str) -> bool:
        """Revoke a token (add to blacklist)."""
        token = self._decode(token_str)
        if token and "jti" in token:
            self._revoked_tokens.add(token["jti"])
            # Limit revocation list size
            if len(self._revoked_tokens) > 10000:
                self._revoked_tokens = set(list(self._revoked_tokens)[-5000:])
            return True
        return False

    # --- Full Authentication Gate ---

    def authenticate(
        self,
        token: str,
        check_injection: bool = True,
        query: str = "",
    ) -> AuthResult:
        """
        Full authentication gate — validate token + check injection.

        Returns AuthResult with authentication status, user info, and allowed namespaces.
        """
        result = AuthResult()

        # 1. Validate token
        jwt_token = self.validate_token(token)
        if jwt_token is None:
            result.error = "Invalid or expired token"
            return result

        result.authenticated = True
        result.user_id = jwt_token.sub
        result.role = jwt_token.role
        result.permissions = jwt_token.permissions
        result.namespaces = jwt_token.namespaces
        result.org_id = jwt_token.org_id

        # 2. Check injection on query
        if check_injection and query:
            injection_result = self._injection_guard.check(query)
            if injection_result.blocked:
                result.authenticated = False
                result.error = f"Injection detected: {injection_result.threat_type}"
                result.injection_detected = True
                result.injection_details = injection_result.details
                logger.warning(
                    f"Injection blocked: user={jwt_token.sub} type={injection_result.threat_type}"
                )

        return result

    # --- Encoding / Decoding (HMAC-SHA256 based JWT-like) ---

    def _encode(self, payload: dict[str, Any]) -> str:
        """Encode payload into a signed token string."""
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": self._algorithm, "typ": "JWT"}).encode()
        ).rstrip(b"=").decode()

        body = base64.urlsafe_b64encode(
            json.dumps(payload, default=str).encode()
        ).rstrip(b"=").decode()

        signature = hmac.new(
            self._secret,
            f"{header}.{body}".encode(),
            hashlib.sha256,
        ).hexdigest()

        return f"{header}.{body}.{signature}"

    def _decode(self, token: str) -> dict[str, Any] | None:
        """Decode and verify a signed token string."""
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, body_b64, signature = parts

        # Verify signature
        expected_sig = hmac.new(
            self._secret,
            f"{header_b64}.{body_b64}".encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            logger.warning("Token signature verification failed")
            return None

        # Decode payload
        try:
            # Add padding
            padding = 4 - len(body_b64) % 4
            body_b64 += "=" * padding
            payload = json.loads(base64.urlsafe_b64decode(body_b64))
            return payload
        except Exception as e:
            logger.error(f"Token decode error: {e}")
            return None


class AuthResult(BaseModel):
    """Result of authentication gate."""

    authenticated: bool = False
    user_id: str = ""
    role: str = ""
    permissions: list[str] = Field(default_factory=list)
    namespaces: list[str] = Field(default_factory=list)
    org_id: str = ""
    error: str = ""
    injection_detected: bool = False
    injection_details: list[str] = Field(default_factory=list)


# --- Injection Guard ---

class InjectionResult(BaseModel):
    """Result of injection detection."""

    blocked: bool = False
    threat_type: str = ""  # prompt_injection, sql_injection, xss, command_injection
    confidence: float = 0.0
    details: list[str] = Field(default_factory=list)
    sanitized_input: str = ""


class InjectionGuard:
    """
    Multi-layer injection detection and prevention.

    Detects:
    1. Prompt injection — attempts to override system instructions
    2. SQL injection — malicious SQL in user input
    3. XSS — cross-site scripting attempts
    4. Command injection — shell command injection
    5. Path traversal — directory traversal attacks

    Every query passes through this guard before reaching agents.
    """

    # Prompt injection patterns
    PROMPT_INJECTION_PATTERNS: list[tuple[str, str]] = [
        # System prompt override attempts
        (r"(?i)ignore\s+(all\s+)?previous\s+(instructions|prompts|rules)", "system_override"),
        (r"(?i)forget\s+(all\s+)?your\s+(instructions|rules|training|programming)", "system_override"),
        (r"(?i)disregard\s+(all\s+)?previous\s+(instructions|context|rules)", "system_override"),
        (r"(?i)you\s+are\s+now\s+(?:a|an)\s+(?!customer|user|student)", "role_hijack"),
        (r"(?i)act\s+as\s+(?:if\s+)?(?:you\s+(?:are|were)\s+)?(?!a\s+customer|a\s+user)", "role_hijack"),
        (r"(?i)new\s+(?:system\s+)?instructions?:", "system_override"),
        (r"(?i)system\s*(?:prompt|message|instruction)\s*:", "system_override"),
        (r"(?i)\[SYSTEM\]", "system_override"),
        (r"(?i)<<\s*(?:SYS|SYSTEM)", "system_override"),
        # Jailbreak patterns
        (r"(?i)(?:DAN|do\s+anything\s+now)\s+mode", "jailbreak"),
        (r"(?i)(?:developer|debug|admin|root|sudo)\s+mode", "jailbreak"),
        (r"(?i)bypass\s+(?:your\s+)?(?:safety|content|ethical)\s+(?:filter|guard|restriction)", "jailbreak"),
        (r"(?i)pretend\s+(?:you\s+)?(?:have\s+)?no\s+(?:rules|restrictions|limits|guidelines)", "jailbreak"),
        # Data exfiltration
        (r"(?i)(?:reveal|show|display|print|output)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions|rules)", "exfiltration"),
        (r"(?i)what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions|rules|guidelines)", "exfiltration"),
        (r"(?i)repeat\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)\s+(?:back|verbatim)", "exfiltration"),
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS: list[str] = [
        r"(?i)(?:'\s*(?:OR|AND)\s+['\d]+=\s*['\d]+)",   # ' OR 1=1
        r"(?i)(?:;\s*(?:DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|EXEC)\s)", # ;DROP TABLE
        r"(?i)(?:UNION\s+(?:ALL\s+)?SELECT)",            # UNION SELECT
        r"(?i)(?:--\s*$)",                                # SQL comment at end
        r"(?i)(?:'\s*;\s*--)",                            # '; --
        r"(?i)(?:(?:0x[0-9a-f]+|x'[0-9a-f]+')\s)",      # hex injection
        r"(?i)(?:WAITFOR\s+DELAY|BENCHMARK\s*\(|SLEEP\s*\()", # timing attacks
        r"(?i)(?:INTO\s+(?:OUT|DUMP)FILE)",               # file ops
    ]

    # XSS patterns
    XSS_PATTERNS: list[str] = [
        r"<script[^>]*>",                    # <script> tags
        r"(?i)javascript\s*:",               # javascript: protocol
        r"(?i)on(?:load|error|click|mouse|focus|blur|submit|change)\s*=",  # event handlers
        r"<iframe[^>]*>",                    # iframe injection
        r"<object[^>]*>",                    # object injection
        r"<embed[^>]*>",                     # embed injection
        r"(?i)data\s*:\s*text/html",         # data URI
        r"(?i)expression\s*\(",              # CSS expression
        r"<svg[^>]*on\w+\s*=",              # SVG with events
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS: list[str] = [
        r";\s*(?:ls|cat|rm|mv|cp|chmod|chown|wget|curl|bash|sh|python|perl|ruby|nc|netcat)\b",
        r"\$\(.*\)",                          # $(command)
        r"`[^`]+`",                           # `command`
        r"\|\s*(?:bash|sh|python|perl|ruby)",  # pipe to shell
        r"(?:&&|\|\|)\s*(?:rm|cat|wget|curl)", # chained commands
        r">\s*/(?:etc|tmp|var|dev)",           # redirect to system paths
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS: list[str] = [
        r"\.\./",                             # ../
        r"\.\.\\",                            # ..\
        r"(?:etc/passwd|etc/shadow|proc/self)", # sensitive files
        r"%2e%2e[/\\]",                       # URL-encoded ../
        r"(?:\.\.%2f|%2e%2e/)",               # mixed encoding
    ]

    def __init__(
        self,
        block_prompt_injection: bool = True,
        block_sql_injection: bool = True,
        block_xss: bool = True,
        block_command_injection: bool = True,
        block_path_traversal: bool = True,
        custom_patterns: list[tuple[str, str]] | None = None,
        whitelist_patterns: list[str] | None = None,
    ):
        self.block_prompt_injection = block_prompt_injection
        self.block_sql_injection = block_sql_injection
        self.block_xss = block_xss
        self.block_command_injection = block_command_injection
        self.block_path_traversal = block_path_traversal
        self._custom_patterns = custom_patterns or []
        self._whitelist = [re.compile(p) for p in (whitelist_patterns or [])]
        self._stats = {"total_checks": 0, "total_blocked": 0, "by_type": {}}

    def check(self, input_text: str) -> InjectionResult:
        """
        Check input for injection attacks.
        Returns InjectionResult with blocking decision.
        """
        self._stats["total_checks"] += 1
        result = InjectionResult(sanitized_input=input_text)
        threats: list[str] = []

        # Check whitelist first
        for pattern in self._whitelist:
            if pattern.match(input_text):
                return result

        # 1. Prompt injection
        if self.block_prompt_injection:
            for pattern, threat_name in self.PROMPT_INJECTION_PATTERNS:
                if re.search(pattern, input_text):
                    threats.append(f"prompt_injection:{threat_name}")

        # 2. SQL injection
        if self.block_sql_injection:
            for pattern in self.SQL_INJECTION_PATTERNS:
                if re.search(pattern, input_text):
                    threats.append("sql_injection")
                    break

        # 3. XSS
        if self.block_xss:
            for pattern in self.XSS_PATTERNS:
                if re.search(pattern, input_text):
                    threats.append("xss")
                    break

        # 4. Command injection
        if self.block_command_injection:
            for pattern in self.COMMAND_INJECTION_PATTERNS:
                if re.search(pattern, input_text):
                    threats.append("command_injection")
                    break

        # 5. Path traversal
        if self.block_path_traversal:
            for pattern in self.PATH_TRAVERSAL_PATTERNS:
                if re.search(pattern, input_text):
                    threats.append("path_traversal")
                    break

        # 6. Custom patterns
        for pattern, threat_name in self._custom_patterns:
            if re.search(pattern, input_text):
                threats.append(f"custom:{threat_name}")

        if threats:
            result.blocked = True
            result.threat_type = threats[0].split(":")[0]
            result.confidence = min(0.5 + 0.15 * len(threats), 1.0)
            result.details = threats
            result.sanitized_input = self.sanitize(input_text)
            self._stats["total_blocked"] += 1
            for t in threats:
                key = t.split(":")[0]
                self._stats["by_type"][key] = self._stats["by_type"].get(key, 0) + 1

        return result

    def sanitize(self, input_text: str) -> str:
        """Remove dangerous content from input while preserving intent."""
        sanitized = input_text

        # Remove script tags
        sanitized = re.sub(r"<script[^>]*>.*?</script>", "", sanitized, flags=re.DOTALL | re.IGNORECASE)
        sanitized = re.sub(r"<(?:iframe|object|embed)[^>]*>", "", sanitized, flags=re.IGNORECASE)

        # Remove SQL injection attempts
        sanitized = re.sub(r"(?i)(?:;\s*(?:DROP|DELETE|ALTER)\s+\w+)", "", sanitized)
        sanitized = re.sub(r"(?i)UNION\s+(?:ALL\s+)?SELECT", "", sanitized)

        # Remove command injection
        sanitized = re.sub(r"\$\([^)]*\)", "", sanitized)
        sanitized = re.sub(r"`[^`]*`", "", sanitized)

        # Remove path traversal
        sanitized = re.sub(r"\.\.[\\/]", "", sanitized)

        # Remove prompt injection markers
        sanitized = re.sub(r"(?i)\[SYSTEM\]", "", sanitized)
        sanitized = re.sub(r"(?i)<<\s*(?:SYS|SYSTEM)", "", sanitized)

        return sanitized.strip()

    def stats(self) -> dict[str, Any]:
        """Get injection guard statistics."""
        return {
            **self._stats,
            "block_rate": (
                self._stats["total_blocked"] / self._stats["total_checks"]
                if self._stats["total_checks"] > 0 else 0.0
            ),
        }


# --- Namespace Scope Manager ---

class NamespaceManager:
    """
    Manages namespace-based access control for vector stores.

    RBAC decides which vector namespaces the user may search.
    This ensures domain restriction — retrieval stays in scope.

    Usage:
        ns = NamespaceManager()
        ns.assign_namespaces("user-1", ["public", "engineering"])
        ns.assign_namespaces("admin-1", ["*"])  # wildcard = all

        allowed = ns.get_allowed_namespaces("user-1")
        filters = ns.build_filters("user-1")  # for vector store
    """

    def __init__(self) -> None:
        self._user_namespaces: dict[str, set[str]] = {}
        self._role_namespaces: dict[str, set[str]] = {
            "admin": {"*"},                   # all namespaces
            "manager": {"public", "internal"},
            "user": {"public"},
            "viewer": {"public"},
            "api": {"public"},
        }
        self._all_namespaces: set[str] = {"public", "internal", "confidential", "restricted"}

    def assign_namespaces(self, user_id: str, namespaces: list[str]) -> None:
        """Assign specific namespaces to a user."""
        self._user_namespaces[user_id] = set(namespaces)

    def get_allowed_namespaces(self, user_id: str, role: str = "user") -> set[str]:
        """Get namespaces a user is allowed to search."""
        # User-specific overrides
        if user_id in self._user_namespaces:
            ns = self._user_namespaces[user_id]
            if "*" in ns:
                return self._all_namespaces.copy()
            return ns

        # Role-based defaults
        role_ns = self._role_namespaces.get(role, {"public"})
        if "*" in role_ns:
            return self._all_namespaces.copy()
        return role_ns

    def register_namespace(self, namespace: str) -> None:
        """Register a new namespace."""
        self._all_namespaces.add(namespace)

    def build_filters(self, user_id: str, role: str = "user") -> dict[str, Any]:
        """Build vector store filters for namespace scoping."""
        allowed = self.get_allowed_namespaces(user_id, role)
        if "*" in allowed or allowed == self._all_namespaces:
            return {}  # No filter needed — user has full access
        return {"namespace": list(allowed)}

    def check_access(self, user_id: str, namespace: str, role: str = "user") -> bool:
        """Check if user can access a specific namespace."""
        allowed = self.get_allowed_namespaces(user_id, role)
        return namespace in allowed or "*" in allowed
