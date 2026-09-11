"""White-box tests WB-07 through WB-12 for the Flask application."""

import base64
import io
import re
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import app as app_module


@pytest.fixture
def client():
    """Return an isolated Flask client and reset the global camera state."""
    app_module.app.config.update(TESTING=True)
    app_module.camera = None
    with app_module.app.test_client() as test_client:
        yield test_client
    app_module.camera = None


def test_wb07_upload_get_renders_empty_form(client):
    """WB-07: GET /upload skips all POST-only statements."""
    response = client.get("/upload")
    assert response.status_code == 200
    assert b"Upload an Image to Analyze" in response.data
    assert b"Detected Emotion" not in response.data
    assert b"data:image/jpeg;base64," not in response.data


def test_wb08_empty_filename_redirects_without_decoding(client):
    """WB-08: an empty filename takes the second predicate's true branch."""
    with patch.object(app_module.cv2, "imdecode") as imdecode:
        response = client.post(
            "/upload",
            data={"image": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/upload")
    imdecode.assert_not_called()


def test_wb09_valid_upload_returns_emotion_and_decodable_jpeg(client):
    """WB-09: a valid upload follows the longest path and returns base64 JPEG."""
    decoded_image = np.zeros((4, 4, 3), dtype=np.uint8)
    processed_image = np.full((4, 4, 3), 127, dtype=np.uint8)
    encoded_jpeg = b"\xff\xd8\xff\xe0test-jpeg"
    with (
        patch.object(app_module.cv2, "imdecode", return_value=decoded_image) as imdecode,
        patch.object(
            app_module,
            "detect_faces_and_emotions",
            return_value=(processed_image, "Happy"),
        ) as detect,
        patch.object(
            app_module.cv2,
            "imencode",
            return_value=(True, np.frombuffer(encoded_jpeg, dtype=np.uint8)),
        ) as imencode,
    ):
        response = client.post(
            "/upload",
            data={"image": (io.BytesIO(b"valid-image-bytes"), "face.jpg")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    assert b"Detected Emotion" in response.data
    assert b"Happy" in response.data
    imdecode.assert_called_once()
    assert imdecode.call_args.args[1] == cv2.IMREAD_COLOR
    detect.assert_called_once_with(decoded_image)
    imencode.assert_called_once_with(".jpg", processed_image)
    match = re.search(rb"data:image/jpeg;base64,([^\"']+)", response.data)
    assert match is not None
    assert base64.b64decode(match.group(1)).startswith(b"\xff\xd8\xff")


def test_wb10_upload_without_image_field_returns_400(client):
    """WB-10: the current implementation exposes its missing-field 400 path."""
    response = client.post("/upload", data={"other": "1"})
    assert response.status_code == 400


def test_wb11_invalid_image_returns_500(client):
    """WB-11: an undecodable image currently reaches processing and returns 500."""
    previous = app_module.app.config.get("PROPAGATE_EXCEPTIONS")
    app_module.app.config["PROPAGATE_EXCEPTIONS"] = False

    def fail_when_image_is_none(image):
        assert image is None
        raise cv2.error("imdecode returned None")

    try:
        with (
            patch.object(app_module.cv2, "imdecode", return_value=None),
            patch.object(
                app_module,
                "detect_faces_and_emotions",
                side_effect=fail_when_image_is_none,
            ) as detect,
        ):
            response = client.post(
                "/upload",
                data={"image": (io.BytesIO(b"not an image"), "fake.jpg")},
                content_type="multipart/form-data",
            )
    finally:
        app_module.app.config["PROPAGATE_EXCEPTIONS"] = previous
    assert response.status_code == 500
    detect.assert_called_once_with(None)


def test_wb12_first_start_creates_camera_and_enables_stream(client):
    """WB-12: the first POST /start creates camera zero exactly once."""
    fake_camera = MagicMock(name="camera")
    with patch.object(
        app_module.cv2, "VideoCapture", return_value=fake_camera
    ) as video_capture:
        response = client.post("/start")
    assert response.status_code == 200
    video_capture.assert_called_once_with(0)
    assert app_module.camera is fake_camera
    assert b"video_feed" in response.data
    assert b"Stop Detection" in response.data


@pytest.mark.xfail(
    strict=True,
    reason="D-02: upload() indexes request.files directly instead of handling a missing image",
)
def test_d02_missing_image_field_should_be_handled_gracefully(client):
    """A missing image field must not expose Flask's default 400 error page."""
    response = client.post("/upload", data={"other": "1"})

    assert response.status_code in {200, 302}
    if response.status_code == 302:
        assert response.headers["Location"].endswith("/upload")


@pytest.mark.xfail(
    strict=True,
    reason="D-01: upload() does not reject the None returned by cv2.imdecode",
)
def test_d01_invalid_image_should_be_rejected_before_detection(client):
    """Invalid bytes must be handled without passing None into face detection."""
    processed_image = np.zeros((2, 2, 3), dtype=np.uint8)

    with (
        patch.object(app_module.cv2, "imdecode", return_value=None),
        patch.object(
            app_module,
            "detect_faces_and_emotions",
            return_value=(processed_image, "unused"),
        ) as detect,
    ):
        response = client.post(
            "/upload",
            data={"image": (io.BytesIO(b"not an image"), "fake.jpg")},
            content_type="multipart/form-data",
        )

    assert response.status_code in {200, 302}
    detect.assert_not_called()


def test_wb13_no_face_skips_model_and_returns_original_image():
    """WB-13: the zero-face branch returns before emotion prediction."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cascade = MagicMock(name="face_cascade")
    cascade.detectMultiScale.return_value = ()
    fake_model = MagicMock(name="model")

    with (
        patch.object(app_module, "face_cascade", cascade),
        patch.object(app_module, "model", fake_model),
    ):
        processed, emotion = app_module.detect_faces_and_emotions(image)

    assert processed is image
    assert emotion == "No face detected"
    cascade.detectMultiScale.assert_called_once()
    _, kwargs = cascade.detectMultiScale.call_args
    assert kwargs == {
        "scaleFactor": 1.1,
        "minNeighbors": 5,
        "minSize": (30, 30),
    }
    fake_model.predict.assert_not_called()


def test_wb14_single_face_is_resized_normalized_and_classified():
    """WB-14: one face follows the complete preprocessing path."""
    image = np.full((80, 80, 3), (10, 20, 30), dtype=np.uint8)
    cascade = MagicMock(name="face_cascade")
    cascade.detectMultiScale.return_value = np.array([[8, 10, 40, 45]])
    fake_model = MagicMock(name="model")
    fake_model.predict.return_value = np.array([[0.05, 0.10, 0.80, 0.05]])

    with (
        patch.object(app_module, "face_cascade", cascade),
        patch.object(app_module, "model", fake_model),
        patch.object(app_module.cv2, "rectangle") as rectangle,
        patch.object(app_module.cv2, "putText") as put_text,
    ):
        processed, emotion = app_module.detect_faces_and_emotions(image)

    assert processed is image
    assert emotion == "Surprise"
    model_input = fake_model.predict.call_args.args[0]
    assert model_input.shape == (1, 96, 96, 3)
    assert model_input.dtype.kind == "f"
    assert np.allclose(model_input[0, 0, 0], np.array([30, 20, 10]) / 255.0)
    rectangle.assert_called_once_with(image, (8, 10), (48, 55), (255, 0, 0), 2)
    assert put_text.call_args.args[1] == "Surprise"
    assert put_text.call_args.args[2] == (8, 0)


def test_wb15_repeated_start_reuses_existing_camera(client):
    """WB-15: the false camera-is-None branch does not reopen device zero."""
    existing_camera = MagicMock(name="existing_camera")
    app_module.camera = existing_camera

    with patch.object(app_module.cv2, "VideoCapture") as video_capture:
        response = client.post("/start")

    assert response.status_code == 200
    assert app_module.camera is existing_camera
    video_capture.assert_not_called()
    assert b"video_feed" in response.data


def test_wb16_video_feed_stops_when_camera_read_fails(client):
    """WB-16: a failed first camera read takes the generator break branch."""
    fake_camera = MagicMock(name="camera")
    fake_camera.read.return_value = (False, None)
    app_module.camera = fake_camera

    with patch.object(app_module, "detect_faces_and_emotions") as detect:
        response = client.get("/video_feed")

    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"
    assert response.data == b""
    fake_camera.read.assert_called_once_with()
    detect.assert_not_called()


def test_wb17_video_feed_yields_encoded_frame_then_stops(client):
    """WB-17: a successful loop iteration yields one multipart JPEG frame."""
    source_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    processed_frame = np.full((4, 4, 3), 127, dtype=np.uint8)
    fake_camera = MagicMock(name="camera")
    fake_camera.read.side_effect = [(True, source_frame), (False, None)]
    app_module.camera = fake_camera
    encoded = np.frombuffer(b"encoded-jpeg", dtype=np.uint8)

    with (
        patch.object(
            app_module,
            "detect_faces_and_emotions",
            return_value=(processed_frame, "Neutral"),
        ) as detect,
        patch.object(app_module.cv2, "imencode", return_value=(True, encoded)) as encode,
    ):
        response = client.get("/video_feed")

    assert response.status_code == 200
    assert response.data == (
        b"--frame\r\nContent-Type: image/jpeg\r\n\r\nencoded-jpeg\r\n"
    )
    detect.assert_called_once_with(source_frame)
    encode.assert_called_once_with(".jpg", processed_frame)
    assert fake_camera.read.call_count == 2


def test_wb18_stop_releases_camera_clears_state_and_redirects(client):
    """WB-18: the true stop branch releases and clears the global camera."""
    fake_camera = MagicMock(name="camera")
    app_module.camera = fake_camera

    response = client.post("/stop")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/real_time")
    fake_camera.release.assert_called_once_with()
    assert app_module.camera is None


@pytest.mark.xfail(
    strict=True,
    reason="D-03: multiple predictions are overwritten; only the final one is returned",
)
def test_d03_multiple_faces_should_return_all_detected_emotions():
    """Every detected face should retain its own emotion result."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cascade = MagicMock(name="face_cascade")
    cascade.detectMultiScale.return_value = np.array(
        [[5, 5, 30, 30], [50, 50, 30, 30]]
    )
    fake_model = MagicMock(name="model")
    fake_model.predict.side_effect = [
        np.array([[0.90, 0.05, 0.03, 0.02]]),
        np.array([[0.05, 0.90, 0.03, 0.02]]),
    ]

    with (
        patch.object(app_module, "face_cascade", cascade),
        patch.object(app_module, "model", fake_model),
    ):
        _, emotions = app_module.detect_faces_and_emotions(image)

    assert emotions == ["Happy", "Sad"]
