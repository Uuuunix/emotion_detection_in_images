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

    

def test_start_creates_camera_once(client):
    with patch('app.cv2.VideoCapture') as vc:
        vc.return_value = MagicMock(isOpened=lambda: True)
        client.post('/start')
        client.post('/start')          # 第二次应走 camera is None 的假分支
        assert vc.call_count == 1      # WB-13 断言


if __name__ == "__main__":
    unittest.main(verbosity=2)