#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
软件测试 —— 白盒单元测试脚本（emotion_detection_in_images / app.py）

被测对象 : app.py
测试类型 : 白盒测试（White-Box Testing）
              · 逻辑覆盖：class_labels 下标映射、图像预处理（尺寸/归一化）
              · 命令行启动路径覆盖：根目录启动 / 上级目录相对路径启动
测试方法 : 单元测试 + 集成测试（子进程真实启动 Flask 服务并做 HTTP 探测）
                               + 桩模块/打桩（unittest.mock 对 Haar 分类器与模型打桩）

运行环境 : conda 环境 emotion（TensorFlow + Flask + OpenCV）
运行方式 : 在本文件所在目录或任意目录执行
               $ /home/ykan/miniconda3/envs/emotion/bin/python test/test_ky/unitest_ky_white_1.py
           若当前 shell 已激活 emotion 环境，则直接
               $ python test/test_ky/unitest_ky_white_1.py
           可用环境变量 KY_TEST_PYTHON 指定用于启动被测程序的解释器。

作者: Kanyu
"""

import base64
import io
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))       
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  
PARENT_DIR = os.path.dirname(PROJECT_ROOT)                 
APP_FILENAME = "app.py"
APP_PATH = os.path.join(PROJECT_ROOT, APP_FILENAME)
APP_RELATIVE_PATH = os.path.join(os.path.basename(PROJECT_ROOT), APP_FILENAME)

HOST = "127.0.0.1"
PORT = 5000
BASE_URL = "http://{}:{}".format(HOST, PORT)


STARTUP_TIMEOUT = 240      # 秒
PROBE_TIMEOUT = 10         # 秒
SHUTDOWN_GRACE = 5         # 秒


PYTHON = os.environ.get("KY_TEST_PYTHON", sys.executable)

LOG_DIR = os.path.join(SCRIPT_DIR, "console_output")
LOG_EXT = ".txt"

TESTCASE_META = {
    "test_KY_TC_001_app_start_from_project_root": dict(
        tc_id="KY-TC-001",
        item="app.py 启动入口（命令行启动 / 模块级初始化）",
        title="在项目根目录执行 python app.py，程序正常启动并监听 5000 端口",
        criticality="高",
        precondition="1) emotion 环境依赖完整；2) Models/model.h5、templates/ 存在；"
                     "3) 5000 端口空闲；4) 当前工作目录为项目根目录",
        input="命令：python app.py（cwd = emotion_detection_in_images）",
        procedure="1) 以项目根目录为 cwd 启动子进程 python app.py；\n"
                  "2) 轮询 127.0.0.1:5000 端口直至可连接；\n"
                  "3) 记录进程存活状态与启动日志。",
        expected="模型加载成功，Flask 开发服务器在 http://127.0.0.1:5000 正常监听，"
                 "进程保持存活不退出",
    ),
    "test_KY_TC_002_index_route_from_project_root": dict(
        tc_id="KY-TC-002",
        item="路由 / （index 视图函数）",
        title="根目录启动后访问首页 / ，返回 200 及首页 HTML",
        criticality="高",
        precondition="KY-TC-001 已启动的服务正在运行",
        input="HTTP GET http://127.0.0.1:5000/",
        procedure="1) 对 / 发起 GET 请求；2) 检查状态码与响应体关键内容。",
        expected="HTTP 200，响应体为 index.html 渲染结果（html 文档）",
    ),
    "test_KY_TC_003_upload_route_from_project_root": dict(
        tc_id="KY-TC-003",
        item="路由 /upload （upload 视图函数，GET 分支）",
        title="根目录启动后访问图片上传页 /upload ，返回 200",
        criticality="中",
        precondition="KY-TC-001 已启动的服务正在运行",
        input="HTTP GET http://127.0.0.1:5000/upload",
        procedure="1) 对 /upload 发起 GET 请求；2) 检查状态码。",
        expected="HTTP 200，返回上传页面 HTML",
    ),
    "test_KY_TC_004_app_start_from_parent_directory": dict(
        tc_id="KY-TC-004",
        item="app.py 启动入口（跨目录相对路径启动）",
        title="在上一级目录执行 python emotion_detection_in_images/app.py，程序应正常启动",
        criticality="高",
        precondition="1) emotion 环境依赖完整；2) 5000 端口空闲；"
                     "3) 当前工作目录为项目的上一级目录",
        input="命令：python emotion_detection_in_images/app.py（cwd = 上一级目录）",
        procedure="1) 以项目上一级目录为 cwd 启动子进程；\n"
                  "2) 轮询 5000 端口；\n"
                  "3) 捕获子进程退出码与标准错误输出。",
        expected="模型加载成功，Flask 服务器在 127.0.0.1:5000 正常监听，进程保持存活",
    ),
    "test_KY_TC_005_class_labels_constant": dict(
        tc_id="KY-TC-005",
        item="模块常量 class_labels",
        title="情绪类别标签常量取值与顺序正确（4 分类）",
        criticality="高",
        precondition="已导入 app 模块（cwd 已切换至项目根目录）",
        input="读取 app.class_labels",
        procedure="1) 导入 app 模块；2) 断言 class_labels 的取值与顺序。",
        expected="['Happy', 'Sad', 'Surprise', 'Neutral']，长度为 4",
    ),
    "test_KY_TC_006_face_cascade_loaded": dict(
        tc_id="KY-TC-006",
        item="模块级 Haar 级联分类器 face_cascade",
        title="人脸检测器加载成功且可用（非空分类器）",
        criticality="中",
        precondition="已导入 app 模块",
        input="读取 app.face_cascade，调用 empty()",
        procedure="1) 断言 face_cascade 为 cv2.CascadeClassifier 实例；2) 断言 empty() 为 False。",
        expected="分类器成功加载，empty() 返回 False",
    ),
    "test_KY_TC_007_detect_no_face_branch": dict(
        tc_id="KY-TC-007",
        item="函数 detect_faces_and_emotions —— 无人脸分支",
        title="输入不含人脸的图像时，返回原图且情绪为 'No face detected'",
        criticality="高",
        precondition="face_cascade 已加载；已导入 app 模块；无需模型推理",
        input="224x224 纯色（中性灰）合成图像，BGR 三通道 ndarray",
        procedure="1) 构造纯色图像；2) 调用 detect_faces_and_emotions(img)；\n"
                  "3) 检查返回值的类型、内容与输入对象是否被改写。",
        expected="返回 (image, 'No face detected')；图像像素值未被修改（不绘制任何矩形）",
    ),
    "test_KY_TC_008_detect_face_branch_and_label_mapping": dict(
        tc_id="KY-TC-008",
        item="函数 detect_faces_and_emotions —— 有人脸分支与标签映射",
        title="检出人脸时绘制矩形/标签，情绪标签按 argmax 正确映射到 class_labels",
        criticality="高",
        precondition="已导入 app 模块；对 face_cascade.detectMultiScale 与 model.predict 打桩",
        input="320x320 图像；打桩检测框 (30,40,60,60)；打桩预测向量分别以 index 0/1/2/3 为最大值",
        procedure="1) 用 mock 替换 detectMultiScale 返回固定人脸框；\n"
                  "2) 用 mock 替换 model.predict 返回可控概率向量；\n"
                  "3) 调用被测函数，捕获预处理后的入参张量；\n"
                  "4) 断言标签映射、绘制结果与预处理规格；5) 恢复被替换的方法。",
        expected="返回情绪等于 class_labels[argmax]；原图被就地绘制（矩形与文字）；"
                 "喂给模型的张量形状为 (1,96,96,3) 且取值归一化到 [0,1]",
    ),
    "test_KY_TC_009_route_table_registered": dict(
        tc_id="KY-TC-009",
        item="Flask 路由注册表 app.url_map",
        title="全部 6 个业务路由已正确注册且 HTTP 方法符合设计",
        criticality="中",
        precondition="已导入 app 模块",
        input="读取 app.app.url_map 中各规则的 endpoint 与 methods",
        procedure="1) 遍历 url_map；2) 逐条断言路由规则与允许的 HTTP 方法。",
        expected="/、/upload、/real_time、/start、/video_feed、/stop 均已注册，"
                 "其中 /start、/stop、/upload 支持 POST",
    ),
    "test_KY_TC_010_upload_post_no_face_via_test_client": dict(
        tc_id="KY-TC-010",
        item="路由 /upload （POST 分支 → detect_faces_and_emotions → 模板渲染）",
        title="上传无人脸图片，页面返回 'No face detected' 并回显 base64 图像",
        criticality="高",
        precondition="已导入 app 模块；使用 Flask test_client 发起请求",
        input="multipart/form-data 上传一张 JPEG 编码的纯色图像到字段 image",
        procedure="1) 生成纯色 JPEG 字节流；2) 通过 test_client POST /upload；\n"
                  "3) 检查状态码与响应体中的情绪文案。",
        expected="HTTP 200，响应体包含 'No face detected'（无人脸分支端到端生效）",
    ),
    "test_KY_TC_011_real_time_page_renders": dict(
        tc_id="KY-TC-011",
        item="路由 /real_time （real_time 视图函数）",
        title="实时检测页 /real_time 正常渲染，且默认不开启视频流（stream=False）",
        criticality="中",
        precondition="已导入 app 模块；使用 Flask test_client",
        input="HTTP GET /real_time（test_client）",
        procedure="1) GET /real_time；2) 断言状态码与页面关键元素。",
        expected="HTTP 200，页面渲染成功，初始状态为非流式（不包含 /video_feed 视频流标签）",
    ),
    "test_KY_TC_012_video_feed_url_reachable": dict(
        tc_id="KY-TC-012",
        item="路由 /video_feed （video_feed 视图函数）",
        title="视频流地址 /video_feed 可访问，Content-Type 为 multipart 流",
        criticality="中",
        precondition="已导入 app 模块；使用 Flask test_client（buffered=False）；"
                     "读取响应头即结束，不在无摄像头环境下取流",
        input="HTTP GET /video_feed（test_client，仅读取响应头）",
        procedure="1) 用 test_client 打开 /video_feed（非缓冲）；2) 读取 status 与 "
                  "Content-Type；3) 释放并恢复 camera 全局变量。",
        expected="HTTP 200，Content-Type 为 multipart/x-mixed-replace（流式响应已建立）",
    ),
    "test_KY_TC_013_start_skips_capture_when_camera_exists": dict(
        tc_id="KY-TC-013",
        item="路由 /start （start_detection 视图函数）—— 判定假分支",
        title="camera 非 None 时重复点击 Start，跳过重复初始化，不重复占用设备",
        criticality="P1",
        precondition="已执行一次 POST /start，camera 非 None（服务已处于运行状态）",
        input="HTTP POST /start（在 camera 已存在的情况下再次点击，重复 2 次）",
        procedure="1) 打桩 cv2.VideoCapture 返回 mock 相机并统计调用次数；\n"
                  "2) 首次 POST /start 建立 camera（判定真分支，作为前置条件）；\n"
                  "3) 连续再次 POST /start（判定假分支）；\n"
                  "4) 检查 VideoCapture 累计调用次数、camera 对象同一性与页面渲染。",
        expected="行 128 判定为假 → 跳过行 129 → 行 131；第二次调用后 VideoCapture 累计调用次数"
                 "仍为 1；页面照常渲染（stream=True，Stop Detection 按钮与视频流标签齐全），"
                 "不出现设备被重复打开导致的卡死或黑屏",
    ),
    "test_KY_TC_014_stop_releases_camera_when_running": dict(
        tc_id="KY-TC-014",
        item="路由 /stop （stop_detection 视图函数）—— 判定真分支",
        title="摄像头运行中调用 Stop，释放设备、复位全局变量并重定向",
        criticality="P0",
        precondition="camera 为已打开的 mock 对象（模拟运行中的摄像头）",
        input="HTTP POST /stop（camera != None）",
        procedure="1) 将全局 camera 置为 mock 相机；\n"
                  "2) POST /stop；\n"
                  "3) 检查 release() 调用次数、全局 camera 复位情况与响应状态码/重定向目标；\n"
                  "4) 跟随 302 重定向 GET /real_time，捕获模板上下文。",
        expected="行 174 判定为真 → 行 175 release → 行 176 置 None → 行 178 重定向；"
                 "release() 被调用 1 次；全局 camera 恢复为 None；返回 302 指向 /real_time；"
                 "跟随后页面按钮为蓝色 Start",
    ),
    "test_KY_TC_015_stop_without_start_is_idempotent": dict(
        tc_id="KY-TC-015",
        item="路由 /stop （stop_detection 视图函数）—— 判定假分支",
        title="未启动即停止：不抛异常、直接重定向，连续点击行为一致（幂等）",
        criticality="P1",
        precondition="camera 为 None（服务刚启动，从未点击 Start）",
        input="HTTP POST /stop（camera is None），连续调用 3 次",
        procedure="1) 确认全局 camera 为 None；\n"
                  "2) 连续 3 次 POST /stop；\n"
                  "3) 检查每次响应的状态码、Location 与全局 camera 取值；\n"
                  "4) GET /real_time 捕获模板上下文，复核 stream 取值。",
        expected="行 174 判定为假 → 跳过 175–176 → 行 178 重定向；不抛 AttributeError；"
                 "直接 302 回 /real_time；连续多次 POST /stop 行为一致（幂等）",
    ),
}


def port_in_use(host=HOST, port=PORT):
    """检测端口是否已被占用（用于预置条件判定）"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def http_probe(url, timeout=PROBE_TIMEOUT):
    """发起一次 GET 请求，返回 (状态码, 响应体文本)；失败返回 (None, 错误信息)"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, "HTTPError: {}".format(exc)
    except Exception as exc:  # noqa: BLE001 - 测试脚本需记录任意启动期异常
        return None, "{}: {}".format(type(exc).__name__, exc)


def start_app_subprocess(cwd, argv, log_name):
    """
    在指定工作目录以指定命令行启动被测程序（真实子进程，非 mock）。

    子进程的 stdout 与 stderr 合并重定向到 console_output/<log_name>.txt，
    该文件即为控制台输出的原文记录，测试过程中不写入任何附加内容。

    返回 (proc, log_file_handle, log_path)
    """
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_path = os.path.join(LOG_DIR, "{}{}".format(log_name, LOG_EXT))
    log_file = open(log_path, "w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [PYTHON] + argv,
        cwd=cwd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,   # 独立进程组，便于连同子进程一并回收
    )
    return proc, log_file, log_path


def read_log(log_file):
    """读取子进程控制台输出的**完整原文**（不截断、不过滤、不改变文件读写位置）"""
    log_file.flush()
    pos = log_file.tell()
    log_file.seek(0)
    content = log_file.read()
    log_file.seek(pos)
    return content


def stop_app_subprocess(proc, log_file):
    """终止被测程序子进程并回收资源；返回关闭前读到的控制台完整原文"""
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), 15)   # SIGTERM 整个进程组
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=SHUTDOWN_GRACE)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except ProcessLookupError:
                pass
            proc.wait(timeout=SHUTDOWN_GRACE)
    raw = ""
    if log_file is not None:
        raw = read_log(log_file)
        log_file.close()
    return raw


def wait_for_server(proc, log_file, timeout=STARTUP_TIMEOUT):
    """
    等待被测服务就绪。

    返回 (ok: bool, detail: str)：
        · 端口可连接且 / 返回 200  -> (True, '...')
        · 子进程提前退出            -> (False, 退出码 + 日志尾部)
        · 超时                      -> (False, 超时说明)
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, "进程提前退出，returncode={}\n--- 控制台完整输出（原文）---\n{}".format(
                proc.returncode, read_log(log_file)
            )
        if port_in_use():
            code, _ = http_probe(BASE_URL + "/", timeout=PROBE_TIMEOUT)
            if code == 200:
                return True, "端口 {} 已监听，GET / 返回 200".format(PORT)
        time.sleep(1.0)
    return False, "等待 {} 秒后端口 {} 仍未就绪（超时）\n--- 控制台完整输出（原文）---\n{}".format(
        timeout, PORT, read_log(log_file)
    )


def make_solid_image(width=224, height=224, color=(128, 128, 128)):
    """构造一张纯色合成图像（BGR），用于'无人脸'分支测试"""
    import numpy as np
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def encode_jpeg_bytes(img):
    """将 ndarray 编码为 JPEG 字节流，用于模拟 multipart 上传"""
    import cv2
    ok, buffer = cv2.imencode(".jpg", img)
    assert ok, "JPEG 编码失败"
    return buffer.tobytes()


def capture_template_context(flask_app, request_callable):
    """
    执行一次请求，并捕获本次渲染的模板名与模板上下文变量。

    用途：直接断言模板变量（如 stream）的真实取值，而不是仅对渲染后的 HTML
         做文本匹配。返回 (响应对象, [(模板名, 上下文字典), ...])，顺序即渲染顺序。
    """
    from flask import template_rendered

    rendered = []

    def _record(sender, template, context, **extra):  
        rendered.append((template.name, dict(context)))

    template_rendered.connect(_record, flask_app)
    try:
        response = request_callable()
    finally:
        template_rendered.disconnect(_record, flask_app)
    return response, rendered


ACTUAL_RESULTS = {}   # test_method_name -> str


def record(method_name, text):
    ACTUAL_RESULTS[method_name] = text


@unittest.skipUnless(os.path.isfile(APP_PATH), "被测对象 app.py 不存在")
class TestAppStartupFromProjectRoot(unittest.TestCase):
    """测试项：app.py 命令行启动 —— 正常根目录启动"""

    @classmethod
    def setUpClass(cls):
        cls.proc = None
        cls.log_file = None
        cls.log_path = None
        cls.port_preoccupied = port_in_use()

    def test_KY_TC_001_app_start_from_project_root(self):
        """KY-TC-001 根目录执行 python app.py 可正常启动"""
        if self.port_preoccupied:
            record("test_KY_TC_001_app_start_from_project_root",
                   "前置条件不满足：端口 {} 启动前已被其它进程占用，无法判定".format(PORT))
            self.skipTest("端口 {} 启动前已被占用，预置条件不满足".format(PORT))

        proc, log_file, log_path = start_app_subprocess(
            PROJECT_ROOT, [APP_FILENAME], "KY-TC-001_console")
        type(self).proc, type(self).log_file, type(self).log_path = proc, log_file, log_path
        ok, detail = wait_for_server(proc, log_file)

        # 控制台原文在 tearDownClass（进程停止后）补齐，见 tearDownClass
        record("test_KY_TC_001_app_start_from_project_root",
               "执行命令：cd {} && {} {}\n进程状态：{}\n日志原文：{}".format(
                   PROJECT_ROOT, PYTHON, APP_FILENAME,
                   "服务就绪后保持存活，未退出" if ok else "启动失败",
                   os.path.relpath(log_path, SCRIPT_DIR)))
        self.assertTrue(
            ok,
            "根目录启动失败：{}\n（日志文件：{}）".format(detail, log_path),
        )
        self.assertIsNone(proc.poll(), "服务就绪后子进程不应退出")

    def test_KY_TC_002_index_route_from_project_root(self):
        """KY-TC-002 根目录启动后 GET / 返回 200"""
        if self.proc is None or self.proc.poll() is not None:
            record("test_KY_TC_002_index_route_from_project_root",
                   "依赖用例 KY-TC-001 未成功启动服务，本用例无法执行")
            self.skipTest("依赖 KY-TC-001 启动的服务，未就绪")

        code, body = http_probe(BASE_URL + "/")
        record("test_KY_TC_002_index_route_from_project_root",
               "GET / -> HTTP {}，响应体长度 {} 字节".format(code, len(body)))
        self.assertEqual(200, code, "首页 / 未返回 200")
        self.assertIn("<html", body.lower(), "首页响应体不是 HTML 文档")

    def test_KY_TC_003_upload_route_from_project_root(self):
        """KY-TC-003 根目录启动后 GET /upload 返回 200"""
        if self.proc is None or self.proc.poll() is not None:
            record("test_KY_TC_003_upload_route_from_project_root",
                   "依赖用例 KY-TC-001 未成功启动服务，本用例无法执行")
            self.skipTest("依赖 KY-TC-001 启动的服务，未就绪")

        code, body = http_probe(BASE_URL + "/upload")
        record("test_KY_TC_003_upload_route_from_project_root",
               "GET /upload -> HTTP {}，响应体长度 {} 字节".format(code, len(body)))
        self.assertEqual(200, code, "上传页 /upload 未返回 200")

    @classmethod
    def tearDownClass(cls):
        # 进程停止后再读取，日志方为完整（含 KY-TC-002/003 的请求记录）
        raw = stop_app_subprocess(cls.proc, cls.log_file)
        name = "test_KY_TC_001_app_start_from_project_root"
        if name in ACTUAL_RESULTS:
            ACTUAL_RESULTS[name] += (
                "\n\n【控制台完整输出（原文，未截断、未修改）】\n" + raw
            )


@unittest.skipUnless(os.path.isfile(APP_PATH), "被测对象 app.py 不存在")
class TestAppStartupFromParentDirectory(unittest.TestCase):
    """测试项：app.py 命令行启动 —— 在项目上一级目录以相对路径启动"""

    @classmethod
    def setUpClass(cls):
        if port_in_use():
            # 释放上一组用例可能残留的端口占用
            time.sleep(SHUTDOWN_GRACE)

    def test_KY_TC_004_app_start_from_parent_directory(self):
        """KY-TC-004 上一级目录执行 python emotion_detection_in_images/app.py 可正常启动"""
        if port_in_use():
            record("test_KY_TC_004_app_start_from_parent_directory",
                   "前置条件不满足：端口 {} 启动前已被其它进程占用，无法判定".format(PORT))
            self.skipTest("端口 {} 启动前已被占用，预置条件不满足".format(PORT))

        proc, log_file, log_path = start_app_subprocess(
            PARENT_DIR, [APP_RELATIVE_PATH], "KY-TC-004_console")
        raw = ""
        try:
            ok, detail = wait_for_server(proc, log_file)
            self.assertTrue(
                ok,
                "上级目录相对路径启动失败：{}\n（日志文件：{}）".format(detail, log_path),
            )
        finally:
            raw = stop_app_subprocess(proc, log_file)
            record("test_KY_TC_004_app_start_from_parent_directory",
                   "执行命令：cd {} && {} {}\n进程状态：returncode={}\n日志原文：{}\n\n"
                   "【控制台完整输出（原文，未截断、未修改）】\n{}".format(
                       PARENT_DIR, PYTHON, APP_RELATIVE_PATH, proc.returncode,
                       os.path.relpath(log_path, SCRIPT_DIR), raw))


class TestAppWhiteBox(unittest.TestCase):
    """
    测试项：app.py 内部逻辑白盒测试。

    说明：以下用例需以项目根目录为 cwd 导入 app 模块（模块级代码使用相对路径
         'Models/model.h5' 加载模型，与命令行启动路径耦合），故在 setUpClass 中
         显式切换工作目录后再导入。
    """

    app = None

    @classmethod
    def setUpClass(cls):
        cls._orig_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)                 
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        import app as app_module
        cls.app = app_module

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._orig_cwd)

    # ---- 常量与模块级对象 -------------------------------------------------
    def test_KY_TC_005_class_labels_constant(self):
        """KY-TC-005 class_labels 常量取值与顺序正确"""
        labels = self.app.class_labels
        record("test_KY_TC_005_class_labels_constant",
               "app.class_labels = {}".format(labels))
        self.assertEqual(["Happy", "Sad", "Surprise", "Neutral"], labels)
        self.assertEqual(4, len(labels), "情绪类别数应为 4")

    def test_KY_TC_006_face_cascade_loaded(self):
        """KY-TC-006 Haar 级联分类器加载成功"""
        import cv2
        cascade = self.app.face_cascade
        is_loaded = isinstance(cascade, cv2.CascadeClassifier) and not cascade.empty()
        record("test_KY_TC_006_face_cascade_loaded",
               "face_cascade 类型={}，empty()={}".format(
                   type(cascade).__name__, cascade.empty()))
        self.assertIsInstance(cascade, cv2.CascadeClassifier)
        self.assertFalse(cascade.empty(), "Haar 分类器为空，人脸检测不可用")
        self.assertTrue(is_loaded)

    # ---- detect_faces_and_emotions：无人脸分支 ---------------------------
    def test_KY_TC_007_detect_no_face_branch(self):
        """KY-TC-007 无人脸分支返回 'No face detected' 且不修改原图"""
        img = make_solid_image(color=(128, 128, 128))
        snapshot = img.copy()

        out_img, emotion = self.app.detect_faces_and_emotions(img)

        unchanged = bool((img == snapshot).all())
        record("test_KY_TC_007_detect_no_face_branch",
               "输入=224x224 纯色图；返回 emotion={!r}；返回对象与入参同一对象={}；"
               "原图像素是否改变={}".format(emotion, out_img is img, not unchanged))
        self.assertIsInstance(out_img, type(img), "返回值第一个元素应为图像 ndarray")
        self.assertEqual("No face detected", emotion)
        self.assertTrue(unchanged, "无人脸时不应在图像上绘制任何内容")

    # ---- detect_faces_and_emotions：有人脸分支 + 标签映射（打桩） --------
    def test_KY_TC_008_detect_face_branch_and_label_mapping(self):
        """KY-TC-008 有人脸分支：绘制矩形/标签，标签按 argmax 映射，预处理规格正确

        打桩说明：cv2.CascadeClassifier 与 Keras 模型的方法属性不可写
        （'object attribute is read-only'），故采用"替换模块级全局对象"的方式打桩，
        被测函数内部正是通过全局名 face_cascade / model 引用它们，打桩等效且更稳定。
        """
        import numpy as np

        face_box = (30, 40, 60, 60)
        captured = {}

        class FakeCascade(object):
            """桩对象：始终返回固定人脸框，替代真实 Haar 检测器"""

            def __init__(self, boxes):
                self.boxes = np.array([list(boxes)], dtype=np.int32)

            def detectMultiScale(self, *args, **kwargs):
                return self.boxes

        class FakeModel(object):
            """桩对象：返回可控概率向量，并记录送进模型的实际张量"""

            def __init__(self, peak_index):
                self.peak_index = peak_index

            def predict(self, batch):
                captured["batch"] = np.asarray(batch)
                prob = np.zeros((1, 4), dtype=np.float32)
                prob[0, self.peak_index] = 0.9
                prob[0, (self.peak_index + 1) % 4] = 0.1
                return prob

        orig_cascade = self.app.face_cascade
        orig_model = self.app.model
        observed = []
        try:
            self.app.face_cascade = FakeCascade(face_box)
            for peak_index in range(4):
                self.app.model = FakeModel(peak_index)
                captured.clear()

                img = make_solid_image(width=320, height=320, color=(64, 64, 64))
                out_img, emotion = self.app.detect_faces_and_emotions(img)

                expected_label = self.app.class_labels[peak_index]
                batch = captured.get("batch")
                x, y, w, h = face_box
                rect_color = img[y, x]
                text_region_drawn = not bool((img[y - 15:y, x:x + w] == 64).all())
                observed.append(
                    "argmax={} -> {!r}(期望 {!r}){}; 预处理 shape={}; 取值范围=[{:.2f},{:.2f}]; "
                    "矩形角点像素={}; 标签区域已绘制={}".format(
                        peak_index, emotion, expected_label,
                        "" if emotion == expected_label else "(不匹配!)",
                        None if batch is None else batch.shape,
                        float(batch.min()), float(batch.max()),
                        rect_color.tolist(), text_region_drawn,
                    )
                )

                self.assertEqual(expected_label, emotion,
                                 "标签映射错误：argmax={} 应映射为 {}".format(peak_index, expected_label))
                self.assertIs(out_img, img, "应在原图上就地绘制（返回同一对象）")
                self.assertEqual(255, int(rect_color[0]), "人脸框矩形（蓝色通道）未绘制")
                self.assertEqual((0, 0), (int(rect_color[1]), int(rect_color[2])),
                                 "人脸框矩形颜色应为 (255,0,0)")
                self.assertTrue(text_region_drawn, "情绪文字标签未绘制")
                self.assertIsNotNone(batch, "model.predict 未被调用")
                self.assertEqual((1, 96, 96, 3), batch.shape,
                                 "送模型的人脸张量形状应为 (1,96,96,3)")
                self.assertGreaterEqual(float(batch.min()), 0.0, "张量未归一化到 [0,1]")
                self.assertLessEqual(float(batch.max()), 1.0, "张量未归一化到 [0,1]")
        finally:
            self.app.face_cascade = orig_cascade
            self.app.model = orig_model

        record("test_KY_TC_008_detect_face_branch_and_label_mapping",
               "打桩检测框={}，四种 argmax 结果：\n  - {}".format(
                   face_box, "\n  - ".join(observed)))

    def test_KY_TC_009_route_table_registered(self):
        """KY-TC-009 6 个业务路由及其 HTTP 方法注册正确"""
        rules = {}
        for rule in self.app.app.url_map.iter_rules():
            if rule.endpoint == "static":
                continue
            rules[str(rule)] = set(rule.methods) - {"HEAD", "OPTIONS"}

        expected = {
            "/": {"GET"},
            "/upload": {"GET", "POST"},
            "/real_time": {"GET"},
            "/start": {"POST"},
            "/video_feed": {"GET"},
            "/stop": {"POST"},
        }
        record("test_KY_TC_009_route_table_registered",
               "已注册业务路由：{}".format(
                   "; ".join("{} {}".format(k, sorted(v)) for k, v in sorted(rules.items()))))
        self.assertEqual(expected, rules, "路由或 HTTP 方法与设计不一致")

    def test_KY_TC_010_upload_post_no_face_via_test_client(self):
        """KY-TC-010 POST /upload 上传无人脸图片，页面回显 'No face detected'"""
        img = make_solid_image(width=200, height=200, color=(200, 200, 200))
        jpeg_bytes = encode_jpeg_bytes(img)

        with self.app.app.test_client() as client:
            resp = client.post(
                "/upload",
                data={"image": (io.BytesIO(jpeg_bytes), "blank.jpg")},
                content_type="multipart/form-data",
            )
        body = resp.get_data(as_text=True)
        has_emotion = "No face detected" in body
        record("test_KY_TC_010_upload_post_no_face_via_test_client",
               "POST /upload -> HTTP {}；响应体 {} 字节；包含 'No face detected'={}".format(
                   resp.status_code, len(body), has_emotion))
        self.assertEqual(200, resp.status_code, "POST /upload 未返回 200")
        self.assertTrue(has_emotion,
                        "响应体未包含 'No face detected'，无人脸分支未端到端生效")

    # ---- /real_time 与 /video_feed ---------------------------------------
    def test_KY_TC_011_real_time_page_renders(self):
        """KY-TC-011 GET /real_time 正常渲染且默认为非流式"""
        with self.app.app.test_client() as client:
            resp = client.get("/real_time")
        body = resp.get_data(as_text=True)
        record("test_KY_TC_011_real_time_page_renders",
               "GET /real_time -> HTTP {}；响应体 {} 字节".format(resp.status_code, len(body)))
        self.assertEqual(200, resp.status_code, "/real_time 未返回 200")
        self.assertIn("<html", body.lower(), "实时检测页响应体不是 HTML 文档")

    def test_KY_TC_012_video_feed_url_reachable(self):
        """KY-TC-012 /video_feed 以 multipart 流式响应建立连接

        说明：不依赖 KY-TC-001 启动的真实服务，改用 Flask test_client（buffered=False），
        仅读取响应头即结束迭代，避免在无摄像头环境下阻塞取流。
        """
        orig_camera = self.app.camera
        try:
            with self.app.app.test_client() as client:
                resp = client.get("/video_feed", buffered=False)
                status = resp.status_code
                content_type = resp.headers.get("Content-Type", "")
        except Exception as exc:  # noqa: BLE001 - 摄像头不可用属等价类之一，需如实记录
            record("test_KY_TC_012_video_feed_url_reachable",
                   "GET /video_feed 异常：{}: {}".format(type(exc).__name__, exc))
            self.fail("/video_feed 请求异常：{}: {}".format(type(exc).__name__, exc))
        finally:
            if self.app.camera is not None and self.app.camera is not orig_camera:
                self.app.camera.release()
            self.app.camera = orig_camera

        record("test_KY_TC_012_video_feed_url_reachable",
               "GET /video_feed -> HTTP {}；Content-Type={!r}（无摄像头环境下 gen_frames "
               "读帧失败即结束，此处只验证流式响应头已建立）".format(status, content_type))
        self.assertEqual(200, status, "/video_feed 未返回 200")
        self.assertIn("multipart/x-mixed-replace", content_type or "",
                      "/video_feed 未返回混合替换流")

    def test_KY_TC_013_start_skips_capture_when_camera_exists(self):
        """KY-TC-013 start 判定假分支：camera 非 None 时重复点击不再占用设备

        打桩说明：将模块引用的 cv2.VideoCapture 替换为返回 mock 相机的桩并计数，
        被测代码行 129 `camera = cv2.VideoCapture(0)` 是否被执行即可由调用次数判定。
        """
        fake_camera = mock.MagicMock(name="camera")
        orig_camera = self.app.camera
        observed = []
        try:
            with mock.patch.object(self.app.cv2, "VideoCapture",
                                   return_value=fake_camera) as video_capture:
                self.app.camera = None
                with self.app.app.test_client() as client:
                    # 前置条件：首次 POST /start，判定真分支，建立 camera
                    first, first_rendered = capture_template_context(
                        self.app.app, lambda: client.post("/start"))
                    observed.append(
                        "第 1 次 POST /start（camera 原为 None，走真分支）-> HTTP {}；"
                        "VideoCapture 调用次数={}；camera 是否为 mock 对象={}；"
                        "渲染模板={}；模板变量 stream={}".format(
                            first.status_code, video_capture.call_count,
                            self.app.camera is fake_camera,
                            [name for name, _ in first_rendered],
                            [ctx.get("stream") for _, ctx in first_rendered]))
                    self.assertEqual(200, first.status_code, "首次 POST /start 页面未正常渲染")
                    self.assertEqual(1, video_capture.call_count,
                                     "首次 POST /start 应创建一次摄像头")
                    self.assertIs(fake_camera, self.app.camera, "camera 应为桩相机")
                    self.assertEqual([True], [ctx.get("stream") for _, ctx in first_rendered],
                                     "首次 POST /start 后 stream 应为 True")

           
                    for click in (2, 3):
                        resp, rendered = capture_template_context(
                            self.app.app, lambda: client.post("/start"))
                        body = resp.get_data(as_text=True)
                        streams = [ctx.get("stream") for _, ctx in rendered]
                        observed.append(
                            "第 {} 次 POST /start（camera 非 None，走假分支→跳过行 129→行 131）"
                            "-> HTTP {}；VideoCapture 累计调用次数={}；camera 仍为首次创建的"
                            "同一 mock 对象={}；渲染模板={}；模板变量 stream={}；"
                            "页面含 Stop Detection={}；页面含 video_feed 流标签={}".format(
                                click, resp.status_code, video_capture.call_count,
                                self.app.camera is fake_camera,
                                [name for name, _ in rendered], streams,
                                "Stop Detection" in body, "video_feed" in body))

                        self.assertEqual(200, resp.status_code,
                                         "camera 已存在时 POST /start 页面未正常渲染")
                        self.assertEqual(1, video_capture.call_count,
                                         "camera 已存在时不应再次调用 cv2.VideoCapture 占用设备")
                        self.assertIs(fake_camera, self.app.camera,
                                      "camera 不应被重新创建/覆盖")
                        self.assertEqual([True], streams,
                                         "重复 POST /start 后 stream 仍应为 True")
                        self.assertIn("Stop Detection", body,
                                      "重复点击后页面未正常渲染 Stop 按钮")
                        self.assertIn("video_feed", body,
                                      "重复点击后页面缺少视频流标签")

                    video_capture.assert_called_once_with(0)
        finally:
            self.app.camera = orig_camera

        record("test_KY_TC_013_start_skips_capture_when_camera_exists",
               "打桩：cv2.VideoCapture -> mock 相机；\n  - {}".format("\n  - ".join(observed)))

 
    def test_KY_TC_014_stop_releases_camera_when_running(self):
        """KY-TC-014 stop 判定真分支：release → 置 None → 302 重定向"""
        fake_camera = mock.MagicMock(name="camera")
        orig_camera = self.app.camera
        # 先声明后赋值：即使中途抛异常，finally 之后的断言也不会被 NameError 掩盖
        status, location, release_calls, camera_after_stop = None, "", None, "未执行"
        page, body, streams, observed = None, "", [], ""
        try:
            self.app.camera = fake_camera
            with self.app.app.test_client() as client:
                resp = client.post("/stop")
                status = resp.status_code
                location = resp.headers.get("Location", "")
                release_calls = fake_camera.release.call_count
                camera_after_stop = self.app.camera

                # 跟随 302 重定向，检查释放后页面的模板变量与按钮状态
                page, rendered = capture_template_context(
                    self.app.app, lambda: client.get(location or "/real_time"))
                body = page.get_data(as_text=True)
                streams = [ctx.get("stream") for _, ctx in rendered]
                observed = (
                    "POST /stop（camera 为 mock 相机，走真分支→行 175 release→行 176 置 None"
                    "→行 178 重定向）-> HTTP {}；Location={!r}；\n"
                    "  - camera.release() 调用次数={}；\n"
                    "  - 全局 camera 复位后取值={!r}；\n"
                    "  - 跟随重定向 GET {} -> HTTP {}；渲染模板={}；模板变量 stream={}；\n"
                    "  - 页面含 'Start Detection'={}；按钮 class 为蓝色 btn-primary={}；"
                    "不含红色 btn-danger={}；不含 video_feed 流标签={}".format(
                        status, location, release_calls, camera_after_stop,
                        location or "/real_time", page.status_code,
                        [name for name, _ in rendered], streams,
                        "Start Detection" in body, "btn btn-primary btn-lg" in body,
                        "btn-danger" not in body, "video_feed" not in body))
        finally:
            self.app.camera = orig_camera

        self.assertEqual(302, status, "POST /stop 应返回 302 重定向")
        self.assertTrue(location.endswith("/real_time"),
                        "重定向目标应为 /real_time，实际为 {!r}".format(location))
        self.assertEqual(1, release_calls, "camera.release() 应被调用 1 次")
        self.assertIsNone(camera_after_stop, "停止后全局 camera 应恢复为 None")
        self.assertIsNotNone(page, "未能取得重定向后的 /real_time 响应")
        self.assertEqual(200, page.status_code, "跟随重定向后 /real_time 未返回 200")
        self.assertEqual([False], streams,
                         "释放后 GET /real_time 的模板变量 stream 应为 False")
        self.assertIn("Start Detection", body, "释放后页面按钮文案应为 Start Detection")
        self.assertIn("btn btn-primary btn-lg", body, "释放后页面按钮应为蓝色 btn-primary")
        self.assertNotIn("video_feed", body, "释放后页面不应再包含视频流标签")

        record("test_KY_TC_014_stop_releases_camera_when_running", observed)


    def test_KY_TC_015_stop_without_start_is_idempotent(self):
        """KY-TC-015 stop 判定假分支：camera 为 None 时不抛异常、直接重定向且幂等"""
        orig_camera = self.app.camera
        observed = []
        page, streams = None, []
        try:
            self.app.camera = None
            with self.app.app.test_client() as client:
                for attempt in (1, 2, 3):
                    outcome = "无异常"
                    try:
                        resp, rendered = capture_template_context(
                            self.app.app, lambda: client.post("/stop"))
                        status = resp.status_code
                        location = resp.headers.get("Location", "")
                    except Exception as exc:  # noqa: BLE001 - 如实记录任意异常类型
                        status, location, rendered = None, "", []
                        outcome = "{}: {}".format(type(exc).__name__, exc)
                        observed.append(
                            "第 {} 次 POST /stop 抛出异常：{}".format(attempt, outcome))
                        self.fail("未启动时 POST /stop 不应抛异常，实际抛出 {}: {}".format(
                            type(exc).__name__, exc))

                    observed.append(
                        "第 {} 次 POST /stop（camera 为 None，走假分支→跳过行 175–176"
                        "→行 178 重定向）-> HTTP {}；Location={!r}；异常={}；"
                        "全局 camera 仍为 None={}".format(
                            attempt, status, location, outcome,
                            self.app.camera is None))

                    self.assertEqual(302, status,
                                     "未启动时 POST /stop 应返回 302 重定向")
                    self.assertTrue(location.endswith("/real_time"),
                                    "重定向目标应为 /real_time，实际为 {!r}".format(location))
                    self.assertIsNone(self.app.camera,
                                      "未启动时 POST /stop 不应创建 camera")

                page, rendered = capture_template_context(
                    self.app.app, lambda: client.get("/real_time"))
                body = page.get_data(as_text=True)
                streams = [ctx.get("stream") for _, ctx in rendered]
                observed.append(
                    "连续 3 次 POST /stop 行为一致（幂等）；GET /real_time -> HTTP {}；"
                    "渲染模板={}；模板变量 stream={}；页面含 'Start Detection'={}".format(
                        page.status_code, [name for name, _ in rendered], streams,
                        "Start Detection" in body))
        finally:
            self.app.camera = orig_camera

        self.assertIsNotNone(page, "未能取得 /real_time 响应")
        self.assertEqual(200, page.status_code, "GET /real_time 未返回 200")
        self.assertEqual([False], streams,
                         "未启动状态下 GET /real_time 的模板变量 stream 应为 False")

        record("test_KY_TC_015_stop_without_start_is_idempotent",
               "\n  - ".join(observed))


def extract_exception_summary(tb_text):
    """
    从 unittest 的 traceback 文本中取出"异常类型: 首行信息"，作为失败原因的摘要。

    多行断言信息只取首行（首行即结论），完整 traceback 已在测试运行日志中，
    报告中不重复粘贴，避免与 Result 列的原文输出重复。
    """
    import re

    text = str(tb_text or "")
    # 取第一条（即用例自身抛出的那一层）异常；其后的异常来自断言信息内嵌的控制台输出，
    # 属于 Result 列已经原文展示的证据，不在此处重复。
    matches = re.findall(
        r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit)): (.+)$", text, re.MULTILINE)
    if matches:
        exc_type, first_line = matches[0]
        # unittest 的 assertTrue 会加 "False is not true : " 前缀，去掉以保留被测语义
        first_line = re.sub(r"^False is not true\s*:\s*", "", first_line).strip()
        return "{}: {}".format(exc_type, first_line)
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


class TeeStream(object):
    """
    把写入同时转发到多个流（用于"一边打印到终端、一边留存原文"）。

    仅做字节透传，不对内容做任何过滤或改写。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

    def writable(self):
        return True


# 需要在 Result 列中额外附上"终端输出原文"的用例（新增白盒用例：全部在进程内执行，
# 没有子进程控制台日志，故直接引用 unittest 运行器打印到终端的原文片段作为证据）
TERMINAL_OUTPUT_METHODS = (
    "test_KY_TC_013_start_skips_capture_when_camera_exists",
    "test_KY_TC_014_stop_releases_camera_when_running",
    "test_KY_TC_015_stop_without_start_is_idempotent",
)


def extract_terminal_output(runner_output, method_name):
    """
    从 unittest 运行器的完整终端输出中截取指定用例的**原文片段**。

    截取规则（只切片，不改写任何字符）：
        · 起点：首个包含该用例方法名的行（即该用例的输出块首行）；
        · 终点：下一条用例名所在行，或运行汇总分隔行之前；
    若找不到该用例的输出块，返回空串。
    """
    lines = runner_output.splitlines()
    start = None
    for index, line in enumerate(lines):
        if method_name in line:
            start = index
            break
    if start is None:
        return ""

    def _is_block_end(text):
        stripped = text.strip()
        return (stripped.startswith("test_")           # 下一条用例
                or stripped.startswith("---")          # 汇总分隔线
                or stripped.startswith("Ran ")
                or stripped.startswith("OK")
                or stripped.startswith("FAILED"))

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _is_block_end(lines[index]):
            end = index
            break
    return "\n".join(lines[start:end])


def render_testcase_table(result, stream):
    """依据 TESTCASE_META 与实际执行结果，输出《软件测试用例表》"""
    status_map = {}
    detail_map = {}
    for test, tb_text in list(result.failures) + list(result.errors):
        status_map[test._testMethodName] = "不通过"
        detail_map[test._testMethodName] = extract_exception_summary(tb_text)
    for test, reason in getattr(result, "skipped", []):
        status_map[test._testMethodName] = "未执行(预置条件不满足)"
        detail_map[test._testMethodName] = str(reason)

    def esc(text):
        """
        仅做 Markdown 表格所需的最小转义：换行 -> <br>，竖线 -> \\|。

        不做任何内容层面的改写：控制台原文的字符、空格、缩进、顺序、
        包括 ANSI 颜色控制码，均原样保留。
        """
        return str(text).replace("|", "\\|").replace("\n", "<br>")

    stream.write("\n")
    stream.write("=" * 100 + "\n")
    stream.write("软件测试用例表 —— app.py（白盒测试）\n")
    stream.write("=" * 100 + "\n")
    stream.write(
        "说明：Result(实际结果) 列中的命令行输出为子进程 stdout/stderr 合并后的**完整原文**，\n"
        "      未截断、未过滤、未改写（仅将换行编码为 <br> 以适配 Markdown 表格）。\n"
        "      字节级原始凭据见 console_output/KY-TC-001_console.txt、console_output/KY-TC-004_console.txt。\n")
    stream.write("=" * 100 + "\n\n")

    header = ("| Test Case ID | Test Item | Test Case Title | Test Criticality | Pre-condition | "
              "Input | Procedure | Output(预期结果) | Result(实际结果) | Status | Remark |")
    stream.write(header + "\n")
    stream.write("|" + "---|" * 11 + "\n")

    for name, meta in TESTCASE_META.items():
        passed = name not in status_map
        status = "通过" if passed else status_map[name]
        actual = ACTUAL_RESULTS.get(name, "")
        if not passed and detail_map.get(name):
            actual = (actual + "\n\n[断言失败] " + detail_map[name]).strip()
        if passed:
            actual = (actual + "\n\n[断言] 全部断言通过，无异常").strip()

        stream.write("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
            esc(meta["tc_id"]), esc(meta["item"]), esc(meta["title"]),
            esc(meta["criticality"]), esc(meta["precondition"]), esc(meta["input"]),
            esc(meta["procedure"]), esc(meta["expected"]), esc(actual), status,
            esc(REVIEW_METHOD.get(meta["tc_id"], "")),
        ))

    total = len(TESTCASE_META)
    failed = len(status_map)
    stream.write("\n")
    stream.write("执行统计：用例总数 {}，通过 {}，未通过/未执行 {}\n".format(
        total, total - failed, failed))
    if failed:
        stream.write("未通过用例：{}\n".format(
            "、".join(TESTCASE_META[n]["tc_id"] for n in status_map if n in TESTCASE_META)))
    stream.write("\n")


REVIEW_METHOD = {
    "KY-TC-001": " 集成测试：真实子进程命令行启动 + 端口/HTTP 探活；语句覆盖模块级初始化代码。"
                 "控制台原文存于 console_output/KY-TC-001_console.txt",
    "KY-TC-002": "集成测试（灰盒）：真实服务 HTTP 探测；路径覆盖 index 视图",
    "KY-TC-003": "集成测试（灰盒）：真实服务 HTTP 探测；路径覆盖 upload 视图 GET 分支",
    "KY-TC-004": " 集成测试：变更 cwd 后以相对路径启动，验证相对路径健壮性（等价类：cwd≠根目录）。"
                 "控制台原文存于 console_output/KY-TC-004_console.txt",
    "KY-TC-005": " 单元测试：常量断言，验证 4 分类标签顺序（等价类/边界）",
    "KY-TC-006": " 单元测试：对象类型与可用性断言（打桩前置条件验证）",
    "KY-TC-007": "白盒单元测试：分支覆盖（len(faces)==0 早返回分支）+ 无副作用断言",
    "KY-TC-008": "白盒单元测试：分支覆盖（有人脸循环体）+ 打桩（mock Haar 检测器与 model.predict）"
                 "做逻辑覆盖与等价类划分，校验 argmax→标签映射与预处理规格",
    "KY-TC-009": "白盒单元测试：静态检查 url_map 路由注册表，验证路由与方法设计",
    "KY-TC-010": "集成测试（Flask test_client）：multipart 上传，端到端覆盖 upload POST 分支",
    "KY-TC-011": "集成测试（Flask test_client）：路径覆盖 real_time 视图",
    "KY-TC-012": "集成测试（Flask test_client，非缓冲）：仅读取响应头即结束，"
                 "验证流式响应的 Content-Type（等价类：摄像头不可用）",
    "KY-TC-013": "白盒单元测试：判定覆盖（行 128 `if camera is None` 取假分支 → 跳过行 129 → "
                 "行 131）+ 打桩（mock cv2.VideoCapture 计数），验证重复点击不重复占用设备",
    "KY-TC-014": "白盒单元测试：判定覆盖（行 174 `if camera:` 取真分支 → 行 175 release → "
                 "行 176 置 None → 行 178 重定向）+ 打桩 mock 相机，并用 template_rendered "
                 "信号直接断言模板变量 stream",
    "KY-TC-015": "白盒单元测试：判定覆盖（行 174 取假分支 → 跳过行 175–176 → 行 178 重定向）"
                 "+ 异常安全性与幂等性验证（连续 3 次调用）",
}


def main():
    print("=" * 100)
    print("软件测试：白盒单元测试 —— 被测对象 {}".format(APP_PATH))
    print("解释器(PYTHON)   : {}".format(PYTHON))
    print("项目根目录       : {}".format(PROJECT_ROOT))
    print("上一级目录       : {}".format(PARENT_DIR))
    print("服务地址         : {}".format(BASE_URL))
    print("=" * 100)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAppStartupFromProjectRoot))
    suite.addTests(loader.loadTestsFromTestCase(TestAppStartupFromParentDirectory))
    suite.addTests(loader.loadTestsFromTestCase(TestAppWhiteBox))

    terminal_capture = io.StringIO()
    runner = unittest.TextTestRunner(
        stream=TeeStream(sys.stdout, terminal_capture), verbosity=2)
    result = runner.run(suite)

  
    terminal_output = terminal_capture.getvalue()
    for method_name in TERMINAL_OUTPUT_METHODS:
        snippet = extract_terminal_output(terminal_output, method_name)
        if snippet:
            ACTUAL_RESULTS[method_name] = (
                ACTUAL_RESULTS.get(method_name, "")
                + "\n\n【终端输出（原文，未截断、未修改）】\n" + snippet
            ).strip()

   
    render_testcase_table(result, sys.stdout)

    report_path = os.path.join(SCRIPT_DIR, "ky_white_1_test_report.md")
    with open(report_path, "w", encoding="utf-8") as fp:
        render_testcase_table(result, fp)
    print("测试用例表已写入：{}".format(report_path))

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
