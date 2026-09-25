
from __future__ import annotations

from unittest.mock import patch

import numpy as np

NORM = 1.0 / 255.0


# --------------------------------------------------------------------------
# EDI-TC-001 / 002  模块级对象
# --------------------------------------------------------------------------
def test_edi_tc_001_model_accepts_documented_input_and_returns_4_classes(subject):
    """EDI-TC-001：真实模型接受 (1,96,96,3) 输入并输出 4 类概率。"""
    assert subject.class_labels == ["Happy", "Sad", "Surprise", "Neutral"]

    batch = np.zeros((1, 96, 96, 3), dtype=np.float32)
    pred = subject.model.predict(batch, verbose=0)

    assert pred.shape == (1, 4), f"输出形状应为 (1,4)，实际 {pred.shape}"
    assert np.isclose(pred.sum(), 1.0, atol=1e-3), "输出应为归一化概率（softmax）"
    assert (pred >= 0).all() and (pred <= 1).all()


def test_edi_tc_002_class_labels_match_training_order(subject):
    """EDI-TC-002：class_labels 顺序必须与训练时 ImageDataGenerator 的 class_indices 一致。

    训练 notebook 中 ``flow_from_directory(classes=['happy','sad','surprise','neutral'])``
    实测得到的 class_indices 为 {'happy':0,'sad':1,'surprise':2,'neutral':3}
    （见测试报告"标签映射核验"一节）。若两者错位，所有预测都会张冠李戴。
    """
    training_class_indices = {"happy": 0, "sad": 1, "surprise": 2, "neutral": 3}
    labels_from_training = [None] * 4
    for name, idx in training_class_indices.items():
        labels_from_training[idx] = name.capitalize()

    assert subject.class_labels == labels_from_training, (
        "class_labels 与训练时的类别索引顺序不一致，预测结果会整体错位"
    )


def test_edi_tc_003_face_cascade_is_loaded(subject):
    """EDI-TC-003：Haar 人脸检测器加载成功且可用（后续用例的前置条件）。"""
    import cv2

    assert isinstance(subject.face_cascade, cv2.CascadeClassifier)
    assert not subject.face_cascade.empty(), "Haar 级联分类器为空，人脸检测不可用"


# --------------------------------------------------------------------------
# EDI-TC-004 / 005  detect_faces_and_emotions 的分支覆盖
# --------------------------------------------------------------------------
def test_edi_tc_004_no_face_early_return_has_no_side_effect(subject, fake_cascade):
    """EDI-TC-004：未检出人脸时早返回，返回入参本身且不改写像素。"""
    image = np.full((224, 224, 3), 128, dtype=np.uint8)
    before = image.copy()

    with patch.object(subject, "face_cascade", fake_cascade([])):
        returned, emotion = subject.detect_faces_and_emotions(image)

    assert emotion == "No face detected"
    assert returned is image, "无人脸分支应原样返回入参对象"
    assert np.array_equal(image, before), "无人脸分支不应修改图像像素"


def test_edi_tc_005_face_branch_draws_and_maps_label(subject, fake_model, fake_cascade):
    """EDI-TC-005：检出人脸时绘制矩形与文字，并按 argmax 映射到 class_labels。"""
    image = np.full((240, 240, 3), 128, dtype=np.uint8)
    before = image.copy()

    for index, expected_label in enumerate(subject.class_labels):
        probs = [0.03, 0.03, 0.03, 0.03]
        probs[index] = 0.91
        model = fake_model(probs)
        with (
            patch.object(subject, "model", model),
            patch.object(subject, "face_cascade", fake_cascade([(40, 60, 100, 100)])),
        ):
            returned, emotion = subject.detect_faces_and_emotions(image.copy())

        assert emotion == expected_label, f"argmax={index} 应映射为 {expected_label}"
        assert returned.shape == before.shape
        # 矩形颜色为 BGR(255,0,0) 纯蓝，绘制后该位置像素必被改写
        assert not np.array_equal(returned, before), "应已绘制检测框"


def test_edi_tc_006_preprocessing_spec_and_channel_order(subject, fake_model, fake_cascade):
    """EDI-TC-006：送模型的张量规格为 (1,96,96,3)、归一化到 [0,1]、通道由 BGR 转为 RGB。"""
    # 人脸区域使用非对称颜色 BGR=(10,200,30)；若未做 BGR2RGB 转换，通道顺序会不同
    bgr = (10, 200, 30)
    image = np.full((240, 240, 3), 128, dtype=np.uint8)
    image[60:160, 40:140] = bgr

    model = fake_model()
    with (
        patch.object(subject, "model", model),
        patch.object(subject, "face_cascade", fake_cascade([(40, 60, 100, 100)])),
    ):
        subject.detect_faces_and_emotions(image)

    tensor = model.captured_inputs[0]
    assert tensor.shape == (1, 96, 96, 3)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    center = tensor[0, 48, 48]
    expected_rgb = np.array([bgr[2], bgr[1], bgr[0]]) * NORM
    assert np.allclose(center, expected_rgb, atol=0.02), (
        f"通道顺序错误：期望 RGB={expected_rgb.round(3)}，实际 {center.round(3)}"
    )


# --------------------------------------------------------------------------
# EDI-TC-007 ~ 008  边界与循环边界
# --------------------------------------------------------------------------
def test_edi_tc_007_face_box_at_origin_does_not_crash(subject, fake_model, fake_cascade):
    """EDI-TC-007：人脸框位于图像左上角 (0,0) 时，裁剪不越界、不抛异常。"""
    image = np.full((240, 240, 3), 128, dtype=np.uint8)
    model = fake_model()
    with (
        patch.object(subject, "model", model),
        patch.object(subject, "face_cascade", fake_cascade([(0, 0, 60, 60)])),
    ):
        _, emotion = subject.detect_faces_and_emotions(image)

    assert emotion == "Happy"
    assert model.captured_inputs[0].shape == (1, 96, 96, 3)


def test_edi_tc_008_image_smaller_than_min_size_no_crash(subject):
    """EDI-TC-008：图像小于 Haar 的 minSize=(30,30) 时不崩溃、返回无人脸。

    使用**真实** Haar 分类器，不打断桩。
    """
    image = np.full((29, 29, 3), 128, dtype=np.uint8)
    returned, emotion = subject.detect_faces_and_emotions(image)
    assert emotion == "No face detected"
    assert returned.shape == image.shape

