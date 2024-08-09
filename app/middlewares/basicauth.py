from fastapi import HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from app.config import settings
import base64
import secrets

# Basic認証のミドルウェアクラス
class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Authorizationヘッダーを取得
        auth_header = request.headers.get("authorization")

        if not auth_header:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Basic"},
                content="Missing Authorization Header",
            )

        scheme, _, credentials = auth_header.partition(' ')
        if scheme.lower() != 'basic':
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Basic"},
                content="Invalid Authentication Scheme",
            )
        try:
            decoded_credentials = secrets.compare_digest(
                base64.b64decode(credentials),
                f"{settings.BASIC_USERNAME}:{settings.BASIC_PASSWORD}".encode("utf-8")
            )
        except Exception:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Basic"},
                content="Invalid credentials format",
            )

        if not decoded_credentials:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Basic"},
                content="Invalid credentials",
            )

        # 次のミドルウェアまたはエンドポイントを呼び出す
        response = await call_next(request)
        return response
