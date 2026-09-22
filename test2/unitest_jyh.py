from __future__ import annotations
import io
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import shutil
import fixed_impl

TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"
STATIC_DIR = TEST_DIR.parent / "static"

TEST_DIR = Path(__file__).resolve().parent
DATA_DIR = TEST_DIR / "data"
REPORT_DIR = TEST_DIR / "reports"
PROJECT_ROOT = TEST_DIR.parent
FER_ROOT = PROJECT_ROOT / "data-balck-box-test" / "test"

CLASS_LABELS = ["Happy", "Sad", "Surprise", "Neutral"]
FER_LABELS = ["happy", "sad", "surprise", "neutral"]
MALFORMED_UPLOADS = ["not_image.jpg", "empty.jpg"]
SAMPLES_PER_CLASS = 10
FACE_SIZE = 224
BENCHMARK_RUNS = 5


def upload(client, name: str, content: bytes):
    return client.post(
        "/upload",
        data={"image": (io.BytesIO(content), name)},
        content_type="multipart/form-data",
    )


def upload_fixture(client, name: str):
    """上传 test/data 中的固定图片，统一测试文件读取方式。"""
    return upload(client, name, (DATA_DIR / name).read_bytes())


def _result_emotion(body: bytes) -> str | None:
    """从上传页 HTML 中解析实际给出的情绪文案。"""
    if b"No face detected" in body:
        return "No face detected"
    for label in CLASS_LABELS:
        if f"Detected Emotion: <strong>{label}</strong>".encode() in body:
            return label
    return None


# --------------------------------------------------------------------------
# EDI-TC-025  异常输入
# --------------------------------------------------------------------------
@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason="交叉验证队友已登记的缺陷（jyh D-01）：cv2.imdecode 对非图片内容返回 None、"
           "对空字节流直接抛 cv2.error，两条路径都没有兜底，生产模式下都是 HTTP 500",
)
@pytest.mark.parametrize("name", MALFORMED_UPLOADS)
def test_edi_tc_025_malformed_upload_does_not_crash(client, subject, name):
    """EDI-TC-025：非图片/空文件内容不应导致服务 500。

    刻意关闭 Flask 的异常传播，让响应形态与**生产模式**一致（返回 500 而非把异常
    抛进测试），这样"缺陷现象"才是用户真正看到的那一个。
    """
    with patch.dict(subject.app.config, {"PROPAGATE_EXCEPTIONS": False}):
        resp = upload_fixture(client, name)
    assert resp.status_code != 500, f"上传 {name} 触发了 HTTP 500"


# --------------------------------------------------------------------------
# EDI-TC-026  格式与尺寸的等价类
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expect_emotion",
    [
        ("gray_mode.png", None),      # 单通道
        ("rgba_mode.png", None),      # 四通道
        ("bitmap.bmp", None),         # 非 JPEG/PNG 编码
        ("corrupt.jpg", None),        # 截断的 JPEG
        ("too_small.png", "No face detected"),   # 小于 minSize
        ("no_face.jpg", "No face detected"),     # 无人脸
    ],
)
def test_edi_tc_026_other_formats_are_handled(client, name, expect_emotion):
    """EDI-TC-026：灰度/带 Alpha/BMP/截断/极小/无人脸等输入均能正常处理且不报错。

    注意 ``corrupt.jpg``（截断到 1/3 的 JPEG）**不会**让服务报错——OpenCV 的
    ``imdecode`` 对截断 JPEG 相当宽容，会返回一张上方可用区域的图。这是本次测试的
    一个认知修正：最初把它归入"畸形输入"是错的，实测证据（见测试报告）推翻了该假设。
    """
    resp = upload_fixture(client, name)
    assert resp.status_code == 200

    emotion = _result_emotion(resp.data)
    assert emotion is not None, "页面既没有情绪标签也没有 'No face detected'"
    if expect_emotion is not None:
        assert emotion == expect_emotion


# --------------------------------------------------------------------------
# EDI-TC-027  多人脸真实数据
# --------------------------------------------------------------------------
def test_edi_tc_027_multi_face_image_returns_one_of_four_labels(client, subject):
    """EDI-TC-027：含 4 张人脸的图片能处理完并给出 4 类之一。

    这里**只记录行为不判定对错**：真正的判定在 EDI-TC-011（缺陷回归 D-02）里。
    该用例的价值在于证明"多人脸场景不会崩，但页面只会显示一个情绪"。
    """
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["_haar检出结果"]["faces_grid.jpg"] == 4, "测试数据应先保证含 4 张人脸"

    resp = upload_fixture(client, "faces_grid.jpg")
    assert resp.status_code == 200
    assert _result_emotion(resp.data) in CLASS_LABELS


# --------------------------------------------------------------------------
# EDI-TC-028  真实测试集抽样
# --------------------------------------------------------------------------
def test_edi_tc_028_sampled_accuracy_on_real_fer_data(client, subject):
    """EDI-TC-028：在真实 FER-2013 测试集抽样上测量端到端检出率与准确率。

    断言只设**保守下界**（检出率 ≥ 0.4、准确率 ≥ 0.5），目的是让用例稳定可重复；
    实测值写入 ``reports/accuracy_metrics.json`` 供测试报告引用。
    """
    samples = []
    for emotion in FER_LABELS:
        files = sorted((FER_ROOT / emotion).glob("*.jpg"))[:SAMPLES_PER_CLASS]
        samples += [(emotion, path) for path in files]
    assert len(samples) == SAMPLES_PER_CLASS * 4

    total = detected = correct = 0
    per_class = {e: {"总数": 0, "检出": 0, "正确": 0} for e in FER_LABELS}

    for truth, path in samples:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)
        resp = upload(client, path.name, cv2.imencode(".jpg", image)[1].tobytes())
        assert resp.status_code == 200

        total += 1
        per_class[truth]["总数"] += 1
        emotion = _result_emotion(resp.data)
        if emotion == "No face detected":
            continue
        detected += 1
        per_class[truth]["检出"] += 1
        if emotion == truth.capitalize():
            correct += 1
            per_class[truth]["正确"] += 1

    assert total > 0, "没有成功读取任何 FER 测试样本"
    detection_rate = detected / total
    accuracy = correct / detected if detected else 0.0

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "accuracy_metrics.json").write_text(json.dumps({
        "样本总数": total,
        "检出人脸数": detected,
        "检出率": round(detection_rate, 4),
        "检出样本上的准确率": round(accuracy, 4),
        "分类别": per_class,
        "测试数据来源": "data-balck-box-test/test（FER-2013 测试集，每类按文件名排序取前 10 张）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    assert detection_rate >= 0.4, f"人脸检出率过低：{detection_rate:.2%}（{detected}/{total}）"
    assert accuracy >= 0.5, f"检出样本上的准确率过低：{accuracy:.2%}"


# --------------------------------------------------------------------------
# EDI-TC-029  性能
# --------------------------------------------------------------------------
@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason="缺陷 D-05：app.py 第 2 行硬编码 CUDA_VISIBLE_DEVICES=-1 强制跑 CPU，"
           "640x480 单帧（含 1 张人脸）实测 222.9 ms ≈ 4.5 FPS，"
           "其中模型推理约 122 ms 是主要开销，达不到实时视频所需的帧率预算",
)
def test_edi_tc_029_single_frame_latency_within_realtime_budget(subject):
    """EDI-TC-029（缺陷回归 D-05）：单帧处理耗时应满足实时预算（≤33.3 ms ≈ 30 FPS）。"""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    frame[100:100 + FACE_SIZE, 200:200 + FACE_SIZE] = cv2.imread(str(DATA_DIR / "face_happy.jpg"))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = subject.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    subject.model.predict(np.zeros((1, 96, 96, 3), np.float32), verbose=0)   # 预热

    if len(faces) > 0:
        subject.detect_faces_and_emotions(frame.copy())
        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            subject.detect_faces_and_emotions(frame.copy())
        elapsed_ms = (time.perf_counter() - start) / BENCHMARK_RUNS * 1000
        detail = f"完整单帧链路（检出 {len(faces)} 张人脸）"
    else:
        face = np.zeros((1, 96, 96, 3), np.float32)
        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            subject.model.predict(face, verbose=0)
        elapsed_ms = (time.perf_counter() - start) / BENCHMARK_RUNS * 1000
        detail = "仅模型推理（本帧未检出人脸，取主导开销）"

    assert elapsed_ms <= 33.3, (
        f"{detail} 实测 {elapsed_ms:.1f} ms/帧，约 {1000 / elapsed_ms:.1f} FPS，"
        "达不到实时视频的帧率预算"
    )


# --------------------------------------------------------------------------
# EDI-TC-030  资源释放
# --------------------------------------------------------------------------
@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason="缺陷 D-06：gen_frames 在取帧失败/客户端断开后直接 return，既不 release 摄像头"
           "也不复位全局 camera，设备会一直被占用直到用户手动点 Stop",
)
def test_edi_tc_030_camera_released_after_stream_ends(client, subject):
    """EDI-TC-030（缺陷回归 D-06）：视频流结束后应释放摄像头设备。"""
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.read.return_value = (False, None)      # 模拟摄像头取帧失败（拔掉/断开）

    with patch.object(subject.cv2, "VideoCapture", return_value=fake_cap):
        client.post("/start")
        resp = client.get("/video_feed", buffered=True)
        resp.close()

    assert fake_cap.release.call_count >= 1, (
        "视频流已结束但摄像头未被释放，设备将一直被占用"
    )


# --------------------------------------------------------------------------
# 修复验证工具函数
# --------------------------------------------------------------------------

def _run_fixed_detect(subject, fake_model, fake_cascade, boxes, image):
    """在修复版 detect_faces_and_emotions 下跑一次，返回 (tensor, 返回值)。"""
    model = fake_model()
    with (
        patch.object(subject, "detect_faces_and_emotions", fixed_impl.detect_faces_and_emotions_fixed),
        patch.object(subject, "model", model),
        patch.object(subject, "face_cascade", fake_cascade(boxes)),
    ):
        returned, emotion = subject.detect_faces_and_emotions(image)
    return model.captured_inputs[0], returned, emotion


# --------------------------------------------------------------------------
# D-01
# --------------------------------------------------------------------------
def test_fix_d01_crop_no_longer_contains_drawn_border(subject, fake_model, fake_cascade):
    """D-01 修复验证：先裁剪后绘制，裁剪区域不再混入边框像素。"""
    image = np.full((240, 240, 3), 128, dtype=np.uint8)
    tensor, _, _ = _run_fixed_detect(subject, fake_model, fake_cascade, [(40, 60, 100, 100)], image)

    face = tensor[0]
    assert not np.all(face[0, :, 2] > 0.99), "第 0 行仍是边框像素"
    assert not np.all(face[:, 0, 2] > 0.99), "第 0 列仍是边框像素"
    # 整张人脸应保持原本的均匀灰色（128/255 ≈ 0.502）
    assert np.allclose(face, 128 / 255.0, atol=0.02), "裁剪区域应全部是人脸像素"


# --------------------------------------------------------------------------
# D-02
# --------------------------------------------------------------------------
def test_fix_d02_multi_face_result_is_order_independent(subject, fake_cascade, multi_face_scene):
    """D-02 修复验证：返回值取面积最大的人脸，与检测框顺序无关。"""
    image, box_big, box_small, make_model = multi_face_scene

    def run(boxes):
        model = make_model()
        with (
            patch.object(subject, "detect_faces_and_emotions",
                         fixed_impl.detect_faces_and_emotions_fixed),
            patch.object(subject, "model", model),
            patch.object(subject, "face_cascade", fake_cascade(boxes)),
        ):
            return subject.detect_faces_and_emotions(image.copy())[1]

    forward, reversed_ = run([box_big, box_small]), run([box_small, box_big])
    assert forward == reversed_ == "Happy", (
        f"修复后应稳定返回面积最大人脸（Happy）的情绪，实际 正序={forward} 逆序={reversed_}"
    )


# --------------------------------------------------------------------------
# D-03
# --------------------------------------------------------------------------
def test_fix_d03_start_reports_failure_when_camera_unavailable(client, subject):
    """D-03 修复验证：摄像头打不开时回到未启动状态，不再假装成功。"""
    dead_cap = MagicMock()
    dead_cap.isOpened.return_value = False
    dead_cap.read.return_value = (False, None)

    original = subject.app.view_functions["start_detection"]
    subject.app.view_functions["start_detection"] = fixed_impl.start_detection_fixed
    try:
        with patch.object(subject.cv2, "VideoCapture", return_value=dead_cap):
            resp = client.post("/start")
    finally:
        subject.app.view_functions["start_detection"] = original

    assert resp.status_code == 200
    assert b"Stop Detection" not in resp.data, "不应展示'运行中'状态"
    assert b"Start Detection" in resp.data
    assert dead_cap.release.called, "打开失败的设备也应被释放"
    assert subject.camera is None, "失败后应复位全局 camera"


# --------------------------------------------------------------------------
# D-04
# --------------------------------------------------------------------------
def test_fix_d04_favicon_served_after_name_fix(client):
    """D-04 修复验证：静态目录存在小写 icon.png 时，模板引用的图标即可正常访问。

    验证方式为临时复制 ``Icon.png -> icon.png``，断言 200 后清理，
    不改变仓库内容（重命名涉及 Git LFS 指针，见 fixed_impl.favicon_fix_hint）。
    """
    source = STATIC_DIR / "Icon.png"
    target = STATIC_DIR / "icon.png"
    assert source.exists(), "前置条件：static/Icon.png 存在"

    created = not target.exists()
    if created:
        shutil.copyfile(source, target)
    try:
        resp = client.get("/static/icon.png")
        assert resp.status_code == 200, "修复后模板引用的 favicon 应可访问"
        assert resp.data[:8] == b"\x89PNG\r\n\x1a\n", "应返回 PNG 内容"
    finally:
        if created:
            target.unlink()


# --------------------------------------------------------------------------
# D-06
# --------------------------------------------------------------------------
def test_fix_d06_stream_releases_camera_on_exit(client, subject):
    """D-06 修复验证：视频流结束（取帧失败/客户端断开）后释放摄像头。"""
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.read.return_value = (False, None)

    original = subject.app.view_functions["video_feed"]
    subject.app.view_functions["video_feed"] = fixed_impl.video_feed_fixed
    try:
        with patch.object(subject.cv2, "VideoCapture", return_value=fake_cap):
            client.post("/start")
            resp = client.get("/video_feed", buffered=True)
            resp.close()
    finally:
        subject.app.view_functions["video_feed"] = original

    assert fake_cap.release.call_count >= 1, "流结束后应释放摄像头"
    assert subject.camera is None, "释放后应复位全局 camera"


# --------------------------------------------------------------------------
# D-05：性能缺陷的诊断验证（无代码级修复，验证"瓶颈定位"是否成立）
# --------------------------------------------------------------------------
def test_fix_d05_latency_is_dominated_by_model_predict(subject):
    """D-05 诊断验证：单帧耗时确实由 model.predict 主导。

    这条用例**不模拟修复**，而是证明"把耗时归因于模型推理"这一结论成立：
    纯预处理（裁剪+缩放+通道转换+归一化）的耗时应当远小于整帧耗时。
    因此该缺陷的修复方向在模型/部署层（换轻量模型、开 GPU、降帧率），
    而不是在视图函数里做微优化——这一点必须写在缺陷报告里，避免给出错误的修复建议。
    """
    image = cv2.imread(str(DATA_DIR / "face_happy.jpg"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = subject.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    assert len(faces) >= 1

    x, y, w, h = faces[0]

    start = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        face = image[y:y + h, x:x + w].copy()
        face = cv2.cvtColor(cv2.resize(face, (96, 96)), cv2.COLOR_BGR2RGB)
        face = np.expand_dims(face.astype(np.float32) / 255.0, axis=0)
    preprocess_ms = (time.perf_counter() - start) / BENCHMARK_RUNS * 1000

    batch = np.zeros((1, 96, 96, 3), np.float32)
    subject.model.predict(batch, verbose=0)
    start = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        subject.model.predict(batch, verbose=0)
    predict_ms = (time.perf_counter() - start) / BENCHMARK_RUNS * 1000

    assert predict_ms > preprocess_ms * 10, (
        f"推理 {predict_ms:.1f} ms vs 预处理 {preprocess_ms:.2f} ms，"
        "若预处理反而更慢则说明瓶颈定位有误，需重写缺陷 D-05 的结论"
    )
