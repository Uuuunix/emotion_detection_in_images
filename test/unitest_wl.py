from unittest.mock import patch, MagicMock
import importlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class WhiteBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 只替换模型加载；使用真实 OpenCV、NumPy、Keras img_to_array。
        # 独立加载被测文件，避免覆盖其他测试正在使用的 app 模块。
        spec = importlib.util.spec_from_file_location("app_whitebox", ROOT / "app.py")
        cls.subject = importlib.util.module_from_spec(spec)
        models = importlib.import_module("tensorflow.keras.models")
        with patch.object(models, "load_model", return_value=Mock()):
            spec.loader.exec_module(cls.subject)

    def setUp(self):
        self.model = Mock()
        self.model.predict.return_value = np.array([[0.90, 0.03, 0.03, 0.04]])
        patcher = patch.object(self.subject, "model", self.model)
        patcher.start()
        self.addCleanup(patcher.stop)

    def faces(self, boxes):
        """固定检测框，保证准确触发目标路径；不模拟图像预处理。"""
        cascade = Mock()
        cascade.detectMultiScale.return_value = np.array(boxes, dtype=np.int32).reshape(-1, 4)
        return patch.object(self.subject, "face_cascade", cascade)

    def test_wb01_one_face(self):
        """WB-01：有脸分支，循环一次，真实绘制蓝框绿字。"""
        image = np.full((240, 240, 3), 128, dtype=np.uint8)
        before = image.copy()
        with self.faces([(40, 60, 100, 100)]), \
                patch.object(cv2, "rectangle", wraps=cv2.rectangle) as rectangle, \
                patch.object(cv2, "putText", wraps=cv2.putText) as text:
            result, emotion = self.subject.detect_faces_and_emotions(image)
        self.assertEqual(result.shape, before.shape)
        self.assertFalse(np.array_equal(result, before))
        self.assertEqual(emotion, "Happy")
        self.model.predict.assert_called_once()
        rectangle.assert_called_once()
        text.assert_called_once()
        self.assertEqual(rectangle.call_args.args[1:], ((40, 60), (140, 160), (255, 0, 0), 2))
        self.assertEqual(text.call_args.args[1:3], ("Happy", (40, 50)))
        self.assertTrue(np.any(np.all(result == (255, 0, 0), axis=-1)))
        self.assertTrue(np.any(np.all(result == (0, 255, 0), axis=-1)))

    def test_wb02_no_face(self):
        """WB-02：真实 Haar 检测纯灰图，无脸时原样提前返回。"""
        image = np.full((512, 512, 3), 128, dtype=np.uint8)
        before = image.copy()
        self.assertFalse(self.subject.face_cascade.empty())
        with patch.object(cv2, "rectangle", wraps=cv2.rectangle) as rectangle, \
                patch.object(cv2, "putText", wraps=cv2.putText) as text:
            result, emotion = self.subject.detect_faces_and_emotions(image)
        self.assertIs(result, image)
        np.testing.assert_array_equal(result, before)
        self.assertEqual(emotion, "No face detected")
        self.model.predict.assert_not_called()
        rectangle.assert_not_called()
        text.assert_not_called()

    def test_wb03_two_faces_last_result(self):
        """WB-03：循环两次；记录 D-03 的现有行为，返回最后一张脸。"""
        image = np.full((240, 360, 3), 128, dtype=np.uint8)
        self.model.predict.side_effect = [
            np.array([[0.90, 0.03, 0.03, 0.04]]),
            np.array([[0.03, 0.90, 0.03, 0.04]]),
        ]
        with self.faces([(30, 60, 100, 100), (200, 60, 100, 100)]), \
                patch.object(cv2, "rectangle", wraps=cv2.rectangle) as rectangle, \
                patch.object(cv2, "putText", wraps=cv2.putText) as text:
            result, emotion = self.subject.detect_faces_and_emotions(image)
        self.assertEqual(self.model.predict.call_count, 2)
        self.assertEqual(rectangle.call_count, 2)
        self.assertEqual(text.call_count, 2)
        self.assertEqual([c.args[1] for c in text.call_args_list], ["Happy", "Sad"])
        self.assertEqual([c.args[1] for c in rectangle.call_args_list], [(30, 60), (200, 60)])
        np.testing.assert_array_equal(result[60, 30], [255, 0, 0])
        np.testing.assert_array_equal(result[60, 200], [255, 0, 0])
        self.assertEqual(emotion, "Sad", "当前接口只返回最后一个情绪，D-03 仍存在")

    def test_wb03_should_return_all_emotions(self):
        """WB-03 补充：按多人结果完整性要求返回列表，当前实现应失败。"""
        image = np.full((240, 360, 3), 128, dtype=np.uint8)
        self.model.predict.side_effect = [
            np.array([[0.90, 0.03, 0.03, 0.04]]),  # 第一张脸 Happy
            np.array([[0.03, 0.90, 0.03, 0.04]]),  # 第二张脸 Sad
        ]
        with self.faces([(30, 60, 100, 100), (200, 60, 100, 100)]):
            _, emotions = self.subject.detect_faces_and_emotions(image)

        self.assertEqual(self.model.predict.call_count, 2)
        # 以列表承载全部结果是 D-03 的修复验收要求，当前字符串接口尚未实现。
        # 不加 expectedFailure，让 unittest 明确报告 FAIL。
        self.assertEqual(
            emotions, ["Happy", "Sad"],
            "D-03：两张脸均已预测，但返回值未保留全部情绪",
        )

    def test_wb04_min_size_contract(self):
        """WB-04：校验 minSize 参数和边界路径；31×31 不保证检出。"""
        scenarios = [(29, [], 0), (30, [(20, 30, 30, 30)], 1), (31, [(20, 30, 31, 31)], 1)]
        for size, boxes, expected_calls in scenarios:
            with self.subTest(mock_face_size=size):
                self.model.predict.reset_mock()
                image = np.full((100, 100, 3), 128, dtype=np.uint8)
                expected_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                with self.faces(boxes) as cascade:
                    _, emotion = self.subject.detect_faces_and_emotions(image)
                cascade.detectMultiScale.assert_called_once()
                np.testing.assert_array_equal(cascade.detectMultiScale.call_args.args[0], expected_gray)
                self.assertEqual(cascade.detectMultiScale.call_args.kwargs,
                                 dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)))
                self.assertEqual(self.model.predict.call_count, expected_calls)
                self.assertEqual(emotion, "Happy" if boxes else "No face detected")
        # 真实检测器：整张图只有 29×29，容不下最小 30×30 的检测框。
        # 上述 mock 不能证明真实 Haar 对 30/31 像素人脸的召回率。
        self.model.predict.reset_mock()
        small = np.full((29, 29, 3), 128, dtype=np.uint8)
        result, emotion = self.subject.detect_faces_and_emotions(small)
        self.assertEqual(emotion, "No face detected")
        np.testing.assert_array_equal(result, np.full((29, 29, 3), 128, dtype=np.uint8))
        self.model.predict.assert_not_called()

    def test_wb05_preprocessing_tensor(self):
        """WB-05：检查真实裁剪、缩放、RGB、float32、归一化和批次维度。"""
        image = np.full((180, 180, 3), 7, dtype=np.uint8)
        # 非灰色、非方形 ROI；水平分色同时校验裁剪位置和 resize。
        image[50:130, 30:90] = (10, 20, 240)
        image[50:130, 90:150] = (230, 40, 30)
        with self.faces([(30, 50, 120, 80)]):
            self.subject.detect_faces_and_emotions(image)
        self.model.predict.assert_called_once()
        tensor = self.model.predict.call_args.args[0]
        self.assertEqual(tensor.shape, (1, 96, 96, 3))
        self.assertEqual(tensor.dtype, np.dtype("float32"))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        np.testing.assert_allclose(tensor[0, 48, 24], np.array([240, 20, 10]) / 255, atol=1e-6)
        np.testing.assert_allclose(tensor[0, 48, 72], np.array([30, 40, 230]) / 255, atol=1e-6)
        # 当前代码先画框再裁剪，ROI 左上角因此变成蓝色（RGB 为 0,0,1）。
        # 这是现有行为记录；张量格式正确不代表模型输入没有被标注污染。
        np.testing.assert_allclose(tensor[0, 0, 0], [0, 0, 1], atol=1e-6)

def test_start_creates_camera_once(client):
    with patch('app.cv2.VideoCapture') as vc:
        vc.return_value = MagicMock(isOpened=lambda: True)
        client.post('/start')
        client.post('/start')          # 第二次应走 camera is None 的假分支
        assert vc.call_count == 1      # WB-13 断言


if __name__ == "__main__":
    unittest.main(verbosity=2)