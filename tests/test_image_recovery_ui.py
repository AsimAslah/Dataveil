from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


def test_restore_ui_uses_automatic_recovery_data() -> None:
    with TestClient(app) as client:
        response = client.get("/imagemasker")

    assert response.status_code == 200
    assert "No upload required" in response.text
    assert "Waiting for protected image" in response.text
    assert 'id="image-recovery-file"' not in response.text
