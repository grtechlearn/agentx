"""Tests for Content Moderation & Vulnerability Scanner."""

import pytest
from agentx.security.moderation import (
    ContentModerator, ModerationConfig, ModerationAction, Severity,
    CategoryConfig, VulnerabilityScanner, VulnerabilityResult,
)


# ---------------------------------------------------------------------------
# ContentModerator Tests
# ---------------------------------------------------------------------------

class TestContentModerator:
    """Tests for the ContentModerator class."""

    def test_clean_text_passes(self):
        mod = ContentModerator()
        result = mod.check("Hello, how can I help you today?")
        assert not result.blocked
        assert result.categories_triggered == []

    def test_profanity_blocked(self):
        mod = ContentModerator()
        result = mod.check("what the fuck is this")
        assert result.blocked
        assert "profanity" in result.categories_triggered

    def test_sexual_content_blocked(self):
        mod = ContentModerator()
        result = mod.check("show me some porn")
        assert result.blocked
        assert "sexual" in result.categories_triggered

    def test_abuse_blocked(self):
        mod = ContentModerator()
        result = mod.check("you are a retard")
        assert result.blocked
        assert "abuse" in result.categories_triggered

    def test_disabled_moderation(self):
        mod = ContentModerator(ModerationConfig(enabled=False))
        result = mod.check("fuck shit damn")
        assert not result.blocked

    def test_category_disable(self):
        config = ModerationConfig(
            categories={
                "profanity": CategoryConfig(enabled=False),
                "sexual": CategoryConfig(enabled=True, action=ModerationAction.BLOCK),
                "abuse": CategoryConfig(enabled=True, action=ModerationAction.BLOCK),
                "violence": CategoryConfig(enabled=False),
                "self_harm": CategoryConfig(enabled=False),
                "drugs": CategoryConfig(enabled=False),
                "custom": CategoryConfig(enabled=False),
            }
        )
        mod = ContentModerator(config)
        # Profanity disabled — should pass
        result = mod.check("what the fuck")
        assert not result.blocked

    def test_redact_action(self):
        config = ModerationConfig(
            categories={
                "profanity": CategoryConfig(enabled=True, action=ModerationAction.REDACT),
                "sexual": CategoryConfig(enabled=False),
                "abuse": CategoryConfig(enabled=False),
                "violence": CategoryConfig(enabled=False),
                "self_harm": CategoryConfig(enabled=False),
                "drugs": CategoryConfig(enabled=False),
                "custom": CategoryConfig(enabled=False),
            }
        )
        mod = ContentModerator(config)
        result = mod.check("what the fuck is this")
        assert not result.blocked
        assert result.action == ModerationAction.REDACT
        assert "fuck" not in result.sanitized_text
        assert "***" in result.sanitized_text

    def test_warn_action(self):
        config = ModerationConfig(
            categories={
                "profanity": CategoryConfig(enabled=True, action=ModerationAction.WARN),
                "sexual": CategoryConfig(enabled=False),
                "abuse": CategoryConfig(enabled=False),
                "violence": CategoryConfig(enabled=False),
                "self_harm": CategoryConfig(enabled=False),
                "drugs": CategoryConfig(enabled=False),
                "custom": CategoryConfig(enabled=False),
            }
        )
        mod = ContentModerator(config)
        result = mod.check("what the shit")
        assert not result.blocked
        assert result.action == ModerationAction.WARN

    def test_custom_blocked_words(self):
        config = ModerationConfig(
            custom_blocked_words=["badword123", "forbidden"],
            categories={
                "profanity": CategoryConfig(enabled=False),
                "sexual": CategoryConfig(enabled=False),
                "abuse": CategoryConfig(enabled=False),
                "violence": CategoryConfig(enabled=False),
                "self_harm": CategoryConfig(enabled=False),
                "drugs": CategoryConfig(enabled=False),
                "custom": CategoryConfig(enabled=False),
            }
        )
        mod = ContentModerator(config)
        result = mod.check("this has badword123 in it")
        assert result.blocked or result.action != ModerationAction.LOG
        assert "badword123" in result.matched_words

    def test_whitelist_overrides(self):
        config = ModerationConfig(
            whitelist_words=["assessment"],
        )
        mod = ContentModerator(config)
        # "ass" is in profanity list but "assessment" is whitelisted
        result = mod.check("the assessment is complete")
        assert not result.blocked

    def test_custom_pattern(self):
        config = ModerationConfig(
            custom_blocked_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],
            categories={
                "profanity": CategoryConfig(enabled=False),
                "sexual": CategoryConfig(enabled=False),
                "abuse": CategoryConfig(enabled=False),
                "violence": CategoryConfig(enabled=False),
                "self_harm": CategoryConfig(enabled=False),
                "drugs": CategoryConfig(enabled=False),
                "custom": CategoryConfig(enabled=False),
            }
        )
        mod = ContentModerator(config)
        result = mod.check("my SSN is 123-45-6789")
        assert "123-45-6789" in result.matched_words

    def test_add_words_dynamically(self):
        mod = ContentModerator()
        mod.add_words("custom", ["newbadword"])
        result = mod.check("this has newbadword")
        assert "newbadword" in result.matched_words

    def test_remove_words_dynamically(self):
        mod = ContentModerator()
        # "damn" is in profanity
        result1 = mod.check("damn it")
        assert result1.blocked
        mod.remove_words("profanity", ["damn"])
        result2 = mod.check("damn it")
        assert not result2.blocked

    def test_stats_tracking(self):
        mod = ContentModerator()
        mod.check("hello")
        mod.check("fuck this")
        stats = mod.stats()
        assert stats["total_checks"] == 2
        assert stats["total_blocked"] == 1

    def test_user_violation_tracking(self):
        config = ModerationConfig(max_violations_before_ban=2)
        mod = ContentModerator(config)
        mod.check("fuck", user_id="user-1")
        assert not mod.is_user_banned("user-1")
        mod.check("shit", user_id="user-1")
        assert mod.is_user_banned("user-1")

    def test_strict_preset(self):
        config = ModerationConfig.strict()
        mod = ContentModerator(config)
        assert config.enabled
        # All categories should be enabled
        for cat in ["profanity", "sexual", "abuse", "violence", "self_harm", "drugs"]:
            assert config.categories[cat].enabled

    def test_permissive_preset(self):
        config = ModerationConfig.permissive()
        mod = ContentModerator(config)
        # Profanity should be disabled
        assert not config.categories["profanity"].enabled
        # Sexual should still be blocked
        assert config.categories["sexual"].enabled

    def test_self_harm_always_critical(self):
        mod = ContentModerator()
        result = mod.check("I want to kill myself")
        assert result.blocked
        assert "self_harm" in result.categories_triggered

    def test_violence_detection(self):
        config = ModerationConfig(
            categories={
                "profanity": CategoryConfig(enabled=False),
                "sexual": CategoryConfig(enabled=False),
                "abuse": CategoryConfig(enabled=False),
                "violence": CategoryConfig(enabled=True, action=ModerationAction.BLOCK),
                "self_harm": CategoryConfig(enabled=False),
                "drugs": CategoryConfig(enabled=False),
                "custom": CategoryConfig(enabled=False),
            }
        )
        mod = ContentModerator(config)
        result = mod.check("I will bomb this place")
        assert result.blocked
        assert "violence" in result.categories_triggered


# ---------------------------------------------------------------------------
# VulnerabilityScanner Tests
# ---------------------------------------------------------------------------

class TestVulnerabilityScanner:
    """Tests for the VulnerabilityScanner class."""

    def test_clean_text_passes(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("Hello, this is a normal message")
        assert not result.has_vulnerabilities
        assert not result.blocked

    def test_detects_eval(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("try using eval('2+2') in Python")
        assert result.has_vulnerabilities
        assert any("eval" in v["description"] for v in result.vulnerabilities)

    def test_detects_api_key(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("my api_key = 'sk-abcdefghijklmnopqrstuvwxyz123456'")
        assert result.has_vulnerabilities
        assert result.blocked  # Critical severity

    def test_detects_openai_key(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("key: sk-aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789abcd")
        assert result.has_vulnerabilities

    def test_detects_aws_key(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("access key AKIAIOSFODNN7EXAMPLE")
        assert result.has_vulnerabilities

    def test_detects_private_key(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("-----BEGIN RSA PRIVATE KEY----- some data")
        assert result.has_vulnerabilities
        assert result.blocked

    def test_detects_javascript_url(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("click here: javascript: alert(1)")
        assert result.has_vulnerabilities

    def test_detects_os_system(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("use os.system('rm -rf /') to clean up")
        assert result.has_vulnerabilities
        assert result.blocked  # Critical

    def test_detects_pickle_load(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("data = pickle.load(open('data.pkl', 'rb'))")
        assert result.has_vulnerabilities

    def test_detects_internal_ip(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("connect to 192.168.1.100 on port 5432")
        assert result.has_vulnerabilities

    def test_detects_ssl_disabled(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("requests.get(url, verify=False)")
        assert result.has_vulnerabilities

    def test_credential_redaction(self):
        scanner = VulnerabilityScanner()
        result = scanner.scan("password = 'mysecretpassword123'")
        assert "[REDACTED]" in result.sanitized_text

    def test_custom_pattern(self):
        scanner = VulnerabilityScanner()
        scanner.add_pattern(r"\bDROP\s+TABLE\b", "SQL DROP TABLE", "critical")
        result = scanner.scan("DROP TABLE users;")
        assert result.has_vulnerabilities

    def test_disabled_checks(self):
        scanner = VulnerabilityScanner(
            check_code_injection=False,
            check_credentials=False,
            check_unsafe_urls=False,
            check_serialization=False,
            check_info_disclosure=False,
            check_insecure_patterns=False,
        )
        result = scanner.scan("eval('dangerous') password='secret123456789abcdef'")
        assert not result.has_vulnerabilities

    def test_block_on_high(self):
        scanner = VulnerabilityScanner(block_on_high=True)
        result = scanner.scan("use eval('2+2') here")
        assert result.blocked  # eval is HIGH severity

    def test_stats_tracking(self):
        scanner = VulnerabilityScanner()
        scanner.scan("hello")
        scanner.scan("eval('bad')")
        stats = scanner.stats()
        assert stats["total_scans"] == 2
        assert stats["total_vulnerabilities"] >= 1


# ---------------------------------------------------------------------------
# Integration: Config presets
# ---------------------------------------------------------------------------

class TestModerationPresets:
    """Test that config presets work correctly."""

    def test_strict_blocks_everything(self):
        mod = ContentModerator(ModerationConfig.strict())
        assert mod.check("fuck").blocked
        assert mod.check("porn").blocked
        assert mod.check("cocaine").blocked

    def test_moderate_selective(self):
        mod = ContentModerator(ModerationConfig.moderate())
        assert mod.check("porn").blocked      # sexual blocked
        assert mod.check("nigger").blocked     # abuse blocked

    def test_permissive_minimal(self):
        mod = ContentModerator(ModerationConfig.permissive())
        result = mod.check("damn it")
        assert not result.blocked  # profanity disabled

    def test_disabled_passes_all(self):
        mod = ContentModerator(ModerationConfig.disabled())
        assert not mod.check("fuck shit porn cocaine").blocked
