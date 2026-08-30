from fastapi.testclient import TestClient

from main import app


def test_image_workspace_exposes_adjustable_face_circle() -> None:
    with TestClient(app) as client:
        response = client.get("/imagemasker")

    assert response.status_code == 200
    assert "Face mask size" in response.text
    assert "Pixel-level semantic face mask" in response.text
    assert 'id="shrink-face-button"' in response.text
    assert 'id="expand-face-button"' in response.text
    assert "dataveil-imagemasker-v4.js" in response.text


def test_cv_uploader_is_visible_and_linked_from_home() -> None:
    with TestClient(app) as client:
        home = client.get("/")
        workspace = client.get("/docmasker")

    assert home.status_code == 200
    assert 'href="/docmasker#upload-cv"' in home.text
    assert workspace.status_code == 200
    assert "Upload CV / document" in workspace.text
    assert "Choose CV, résumé, or document" in workspace.text
    assert 'accept=".pdf,.docx,.pptx' in workspace.text
    assert "dataveil-docmasker-v2.js" in workspace.text


def test_all_pages_use_the_shared_accessible_design_system() -> None:
    with TestClient(app) as client:
        pages = [
            client.get("/"),
            client.get("/docmasker"),
            client.get("/imagemasker"),
        ]

    for response in pages:
        assert response.status_code == 200
        assert "dataveil-design-system.css" in response.text
        assert "dataveil-ui.js" in response.text
        assert 'aria-label="Primary navigation"' in response.text
        assert 'class="process-trail"' in response.text

    assert 'role="tab"' in pages[1].text
    assert 'data-confirm-dialog' in pages[1].text
    assert 'data-drop-zone' in pages[1].text
    assert 'role="tab"' in pages[2].text
    assert 'data-confirm-dialog' in pages[2].text
    assert 'data-drop-zone' in pages[2].text
