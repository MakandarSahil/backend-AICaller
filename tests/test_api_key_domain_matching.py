import os


os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test")
os.environ.setdefault("AZURE_SPEECH_KEY", "test")
os.environ.setdefault("AZURE_SPEECH_REGION", "eastus")
os.environ.setdefault("GROQ_API_KEY", "test")

from app.middleware.auth import _is_domain_allowed


def test_wildcard_requires_subdomain_boundary():
    assert _is_domain_allowed("https://app.example.com", ["*.example.com"])
    assert not _is_domain_allowed("https://badexample.com", ["*.example.com"])


def test_domain_matching_is_case_insensitive():
    assert _is_domain_allowed("https://Example.COM", ["example.com"])
