from __future__ import annotations

from main import app


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
