import time
import datetime
import jwt
import httpx

class GitHubAuth:
    def __init__(self, app_id: str, private_key: str, installation_id: int):
        self.app_id = app_id
        self.private_key = private_key
        self.installation_id = installation_id
        self.installation_token: str | None = None
        self.installation_expires_at: float | None = None

    def generate_jwt(self) -> str:
        now = int(time.time())
        payload = {
            "iat": now - 10,
            "exp": now + 600,
            "iss": self.app_id
        }
        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        return token if isinstance(token, str) else token.decode("utf-8")

    async def generate_installation_token(self) -> str:
        jwt_token = self.generate_jwt()
        url = f"https://api.github.com/app/installations/{self.installation_id}/access_tokens"
        headers = {"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"}

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            token = data.get("token")
            expires_at = data.get("expires_at")
            dt = datetime.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            self.installation_expires_at = dt.timestamp()
            self.installation_token = token
            return token

    async def ensure_token(self) -> str:
        if self.installation_token and self.installation_expires_at:
            if time.time() < (self.installation_expires_at - 30):
                return self.installation_token
        return await self.generate_installation_token()


def verify_signature(body_bytes: bytes, x_hub_signature_256: str | None):
    """Verify webhook HMAC signature using WEBHOOK_SECRET from settings.
    Raises HTTPException on failure.
    """
    from fastapi import HTTPException
    import hmac
    import hashlib
    from config.settings import settings

    secret = settings.WEBHOOK_SECRET
    if not secret:
        return
    if not x_hub_signature_256:
        raise HTTPException(status_code=401, detail="Missing X-Hub-Signature-256 header")
    try:
        expected = "sha256=" + hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
    except Exception:
        raise HTTPException(status_code=500, detail="Error computing HMAC")
    if not hmac.compare_digest(expected, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")
