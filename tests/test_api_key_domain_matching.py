import os

import pytest

_REQUIRED_ENV = {
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test",
    "AZURE_SPEECH_KEY": "test",
    "AZURE_SPEECH_REGION": "eastus",
    "GROQ_API_KEY": "test",
}
_PREEXISTING_ENV = {key: os.environ.get(key) for key in _REQUIRED_ENV}

for key, value in _REQUIRED_ENV.items():
    os.environ.setdefault(key, value)

from app.middleware.auth import _is_domain_allowed


@pytest.fixture(scope="module", autouse=True)
def _restore_env_after_module():
    yield
    for key, previous_value in _PREEXISTING_ENV.items():
        if previous_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous_value


def test_wildcard_requires_subdomain_boundary():
    assert _is_domain_allowed("https://app.example.com", ["*.example.com"])
    assert not _is_domain_allowed("https://badexample.com", ["*.example.com"])


def test_domain_matching_is_case_insensitive():
    assert _is_domain_allowed("https://Example.COM", ["example.com"])
