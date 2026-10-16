from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Tools for StudyMind AI": "API is live!"}


def test_favicon_redirect() -> None:
    response = client.get("/favicon.ico", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/static/favicon.png"


def test_static_favicon() -> None:
    response = client.get("/static/favicon.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content
