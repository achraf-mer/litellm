#!/usr/bin/env python3
"""
Per-deployment OAuth2 client_credentials (private_key_jwt) token, sent upstream as the deployment's api_key.

    litellm_params:
      model: openai/<model>
      api_base: https://gateway.example.com/v1
      api_key: unused
      client_cert: /certs/gateway/tls.crt
      client_key: /certs/gateway/tls.key
      ssl_verify: /certs/gateway/ca.crt
      h2o_oauth:
        token_url: https://idp.example.com/oauth2/token
        client_id: my-client
        client_private_key: os.environ/GATEWAY_SIGNING_KEY
        assertion_alg: ES256
        scope: optional-scope
        client_cert: /certs/idp/tls.crt
        client_key: /certs/idp/tls.key
        ssl_verify: /certs/idp/ca.crt

`client_private_key` is PEM text, `os.environ/NAME` or `file:///path`. Every TLS field and `scope` are optional
"""

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import httpx
import jwt
from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import get_client_cert_ssl_context, get_ssl_configuration
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import CallTypes

CONFIG_KEY = "h2o_oauth"
REFRESH_BEFORE_EXPIRY_SEC = 30.0
DEFAULT_EXPIRES_IN_SEC = 300.0
ASSERTION_TTL_SEC = 60
CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"


class OAuthConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    token_url: str
    client_id: str
    client_private_key: str
    assertion_alg: str = "ES256"
    scope: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    ssl_verify: Optional[Union[bool, str]] = None
    timeout: float = 30.0


class _TokenResponse(BaseModel):
    access_token: str
    expires_in: Optional[float] = None


@dataclass(frozen=True, slots=True)
class _Token:
    value: str
    refresh_at: float


def _resolve_secret(ref: str) -> str:
    if ref.startswith("os.environ/"):
        value = get_secret_str(ref)
        if not value:
            raise ValueError(f"client_private_key references {ref}, which is unset")
        return value
    if ref.startswith("file://"):
        return Path(ref[len("file://") :]).read_text()
    return ref


def _describe_validation_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
        for err in error.errors(include_url=False, include_input=False)
    )


def _auth_error(message: str, model: str) -> litellm.AuthenticationError:
    return litellm.AuthenticationError(message=message, llm_provider="h2o_oauth", model=model)


class OAuthAuthHook(CustomLogger):
    def __init__(
        self,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__()
        self._transport = transport
        self._clock = clock
        self._tokens: Dict[str, _Token] = {}
        self._inflight: Dict[str, "asyncio.Task[_Token]"] = {}

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, object], call_type: Optional[CallTypes]
    ) -> Optional[Dict[str, object]]:
        raw_config = kwargs.get(CONFIG_KEY)
        if raw_config is None:
            return None
        model = str(kwargs.get("model", ""))
        try:
            config = OAuthConfig.model_validate(raw_config)
        except ValidationError as e:
            raise _auth_error(f"invalid h2o_oauth config: {_describe_validation_error(e)}", model) from None
        try:
            token = await self._get_token(config)
        except ValidationError as e:
            raise _auth_error(f"unexpected token endpoint response: {_describe_validation_error(e)}", model) from None
        except Exception as e:
            raise _auth_error(f"could not obtain an h2o_oauth token: {type(e).__name__}: {e}", model) from None
        return {**{k: v for k, v in kwargs.items() if k != CONFIG_KEY}, "api_key": token}

    async def _get_token(self, config: OAuthConfig) -> str:
        key = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
        cached = self._tokens.get(key)
        if cached is not None and self._clock() < cached.refresh_at:
            return cached.value
        task = self._inflight.get(key)
        if task is None or task.get_loop() is not asyncio.get_running_loop():
            task = asyncio.ensure_future(self._refresh(key, config))
            self._inflight[key] = task
        return (await asyncio.shield(task)).value

    async def _refresh(self, key: str, config: OAuthConfig) -> _Token:
        try:
            token = await self._fetch(config)
            self._tokens[key] = token
            return token
        finally:
            self._inflight.pop(key, None)

    async def _fetch(self, config: OAuthConfig) -> _Token:
        now = self._clock()
        assertion = jwt.encode(
            {
                "iss": config.client_id,
                "sub": config.client_id,
                "aud": config.token_url,
                "iat": int(now),
                "exp": int(now) + ASSERTION_TTL_SEC,
                "jti": str(uuid.uuid4()),
            },
            _resolve_secret(config.client_private_key),
            algorithm=config.assertion_alg,
        )
        form = {
            "grant_type": "client_credentials",
            "client_id": config.client_id,
            "client_assertion_type": CLIENT_ASSERTION_TYPE,
            "client_assertion": assertion,
            **({"scope": config.scope} if config.scope else {}),
        }
        verify = (
            get_client_cert_ssl_context(config.ssl_verify, config.client_cert, config.client_key)
            if config.client_cert
            else get_ssl_configuration(config.ssl_verify)
        )
        async with httpx.AsyncClient(verify=verify, timeout=config.timeout, transport=self._transport) as client:
            response = await client.post(config.token_url, data=form)
        if response.status_code != 200:
            raise ValueError(f"token endpoint returned {response.status_code}: {response.text[:200]}")
        body = _TokenResponse.model_validate_json(response.content)
        expires_in = body.expires_in or DEFAULT_EXPIRES_IN_SEC
        return _Token(
            value=body.access_token,
            refresh_at=now + max(expires_in - REFRESH_BEFORE_EXPIRY_SEC, expires_in / 2),
        )


oauth_auth_hook = OAuthAuthHook()
