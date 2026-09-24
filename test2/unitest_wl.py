#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EDI-TC-012～EDI-TC-023 路由接口测试（unittest 形式）。

测试通过 Flask ``test_client`` 在进程内调用真实视图函数。图片上传用例保留
真实 Haar 检测、真实模型推理和模板渲染；摄像头相关用例使用可控的测试替身。
"""

from __future__ import annotations

import base64
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
JPEG_MAGIC = b"\xff\xd8\xff"


class RouteIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """独立加载应用一次，供 12 条用例共用真实模型。"""
        spec = importlib.util.spec_from_file_location("app_wl_route", ROOT / "app.py")
        cls.subject = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.subject)
        cls.subject.app.config.update(TESTING=True)

    def setUp(self):
        self.subject.camera = None
        self.client_context = self.subject.app.test_client()
        self.client = self.client_context.__enter__()

    def tearDown(self):
        camera = self.subject.camera
        if camera is not None:
            camera.release()
        self.subject.camera = None
        self.client_context.__exit__(None, None, None)

    def upload(self, filename: str, content: bytes, field: str = "image"):
        """以 multipart/form-data 提交一个文件字段。"""
        return self.client.post(
            "/upload",
            data={field: (io.BytesIO(content), filename)},
            content_type="multipart/form-data",
        )

    def test_edi_tc_012_index_page(self):
        """EDI-TC-012：首页返回 200 并包含两个入口链接。"""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Emotion Detection App", response.data)
        self.assertIn(b"/upload", response.data)
        self.assertIn(b"/real_time", response.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
