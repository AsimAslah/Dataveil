from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from main import RequestBodyLimitMiddleware, app


def test_both_routers_are_loaded_in_the_same_application() -> None:
    expected_routes = {
        "docmasker": "/docmasker",
        "imagemasker": "/imagemasker",
        "mask_text": "/api/doc/mask",
        "restore_masked_text": "/api/doc/restore",
        "detect_image": "/api/image/detect",
        "mask_image": "/api/image/mask",
        "restore_image": "/api/image/restore",
    }

    assert {
        name: str(app.url_path_for(name))
        for name in expected_routes
    } == expected_routes


def test_only_root_module_owns_a_fastapi_application() -> None:
    import DocMasker.main as doc_entry

    assert not hasattr(doc_entry, "app")
    assert app.title == "DataVeil"


def test_streamed_body_limit_does_not_reflect_private_content() -> None:
    limited_app = FastAPI()
    limited_app.add_middleware(RequestBodyLimitMiddleware, max_body_size=16)

    @limited_app.post("/consume")
    async def consume(request: Request):
        return {"bytes": len(await request.body())}

    def body_chunks():
        yield b"SENSITIVE_VALUE"
        yield b"-more-private-data"

    with TestClient(limited_app) as client:
        response = client.post(
            "/consume",
            content=body_chunks(),
            headers={"Content-Type": "application/octet-stream"},
        )

    assert response.status_code == 413
    assert "SENSITIVE_VALUE" not in response.text
    assert response.json()["detail"] == "The request is too large."
