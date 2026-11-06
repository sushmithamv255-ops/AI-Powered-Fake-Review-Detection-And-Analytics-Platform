import io

from fake_review_detector.app import app


def test_upload_limit_is_above_16mb():
    assert app.config.get("MAX_CONTENT_LENGTH", 0) > 16 * 1024 * 1024


def test_large_upload_returns_413():
    client = app.test_client()
    payload = b"A" * (17 * 1024 * 1024)
    response = client.post(
        "/upload",
        data={"dataset_file": (io.BytesIO(payload), "large.csv")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    assert response.status_code == 413
