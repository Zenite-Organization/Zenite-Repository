import asyncio
import time
import unittest

from clients.github.github_auth import GitHubAuth


class TestGitHubAuth(unittest.TestCase):
    def test_reuses_cached_token_when_not_expired(self):
        auth = GitHubAuth("app", "key", 1)
        auth.installation_token = "cached-token"
        auth.installation_expires_at = time.time() + 300

        token = asyncio.run(auth.ensure_token())
        self.assertEqual(token, "cached-token")

    def test_renews_token_when_expired(self):
        auth = GitHubAuth("app", "key", 1)
        auth.installation_token = "old-token"
        auth.installation_expires_at = time.time() - 10

        async def fake_generate_installation_token():
            return "new-token"

        auth.generate_installation_token = fake_generate_installation_token
        token = asyncio.run(auth.ensure_token())
        self.assertEqual(token, "new-token")


if __name__ == "__main__":
    unittest.main()
