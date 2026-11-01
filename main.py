"""Unified FastAPI entry point for the local-only DataVeil application."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from DocMasker.router import router as docmasker_router
from ImageMasker.router import router as imagemasker_router


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 64 * 1024 * 1024


class RequestBodyTooLarge(Exception):
    """Internal signal used before request data reaches an endpoint."""


class RequestBodyLimitMiddleware:
    """Enforce a byte limit even when Content-Length is absent or untrusted."""

    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                await self._error(
                    scope,
                    receive,
                    send,
                    400,
                    "The request metadata is invalid.",
                )
                return
            if declared_size < 0:
                await self._error(
                    scope,
                    receive,
                    send,
                    400,
                    "The request metadata is invalid.",
                )
                return
            if declared_size > self.max_body_size:
                await self._error(
                    scope, receive, send, 413, "The request is too large."
                )
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_size:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await self._error(
                scope, receive, send, 413, "The request is too large."
            )

    @staticmethod
    async def _error(
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        detail: str,
    ) -> None:
        response = JSONResponse(
            status_code=status_code,
            content={"detail": detail},
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )
        await response(scope, receive, send)

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

app = FastAPI(
    title="DataVeil",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_size=MAX_REQUEST_BYTES,
)

# Specific static mounts must be registered before the shared /static mount.
app.mount(
    "/static/docmasker",
    StaticFiles(directory=str(PROJECT_ROOT / "DocMasker" / "static")),
    name="docmasker_static",
)
app.mount(
    "/static/imagemasker",
    StaticFiles(directory=str(PROJECT_ROOT / "ImageMasker" / "static")),
    name="imagemasker_static",
)
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "static")),
    name="root_static",
)

app.include_router(docmasker_router)
app.include_router(imagemasker_router)


@app.middleware("http")
async def privacy_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self'; script-src 'self'; "
        "img-src 'self' data: blob:; connect-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    )
    return response


@app.exception_handler(RequestValidationError)
async def safe_validation_error(
    _request: Request, _exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request data is invalid or exceeds the allowed size."
        },
    )


@app.exception_handler(Exception)
async def safe_unhandled_error(
    _request: Request, _exc: Exception
) -> JSONResponse:
    # Privacy-sensitive values are deliberately not reflected or logged.
    return JSONResponse(
        status_code=500,
        content={"detail": "DataVeil could not process this request."},
    )


@app.get("/", response_class=HTMLResponse, name="home")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "DataVeil",
        "version": "1.0.0",
        "local_only": True,
        "routers": ["docmasker", "imagemasker"],
    }
