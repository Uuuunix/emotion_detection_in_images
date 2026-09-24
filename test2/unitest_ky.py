
from __future__ import annotations

from unittest.mock import patch

import numpy as np


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
# EDI-TC-004  detect_faces_and_emotions 的无人脸分支
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
