import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anime_avatar_runtime.api import create_app
from anime_avatar_runtime.state import MAX_IMAGE_BYTES, initialize, load_state


def reference() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (32, 48), (80, 120, 160)).save(stream, format="PNG")
    return stream.getvalue()


def test_initialize_survives_restart_and_refuses_replacement(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/avatar/state").status_code == 404
        response = client.post("/avatar/init", content=reference())
        assert response.status_code == 201
        state = response.json()
        assert (state["width"], state["height"], state["mode"]) == (32, 48, "paused")
        assert client.post("/avatar/init", content=reference()).status_code == 409
        assert client.get("/runtime/status").json()["autonomous_loop"] is False
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.get("/avatar/state").json() == state


@pytest.mark.parametrize("content", [b"", b"not an image"])
def test_invalid_reference_does_not_create_checkpoint(tmp_path: Path, content: bytes) -> None:
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/avatar/init", content=content).status_code == 422
    assert not (tmp_path / "state.json").exists()


def test_large_upload_is_rejected_before_decoding(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/avatar/init", content=b"x" * (MAX_IMAGE_BYTES + 1)).status_code == 413
    assert not (tmp_path / "state.json").exists()


def test_reference_tampering_is_not_silently_accepted(tmp_path: Path) -> None:
    state = initialize(tmp_path, reference())
    (tmp_path / "references" / f"{state.reference_sha256}.png").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="SHA-256"):
        create_app(tmp_path)


def test_invalid_checkpoint_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_state(tmp_path)
    with pytest.raises(FileExistsError):
        initialize(tmp_path, reference())
