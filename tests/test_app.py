from __future__ import annotations

from main import app


def test_both_routers_are_loaded_in_the_same_application() -> None:
    paths = {route.path for route in app.routes}

    assert {
        "/docmasker",
        "/imagemasker",
        "/api/doc/mask",
        "/api/doc/restore",
        "/api/image/detect",
        "/api/image/mask",
        "/api/image/restore",
    } <= paths


def test_only_root_module_owns_a_fastapi_application() -> None:
    import DocMasker.main as doc_entry

    assert not hasattr(doc_entry, "app")
    assert app.title == "DataVeil"
