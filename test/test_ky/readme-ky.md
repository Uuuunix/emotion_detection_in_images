# 白盒测试用例集（test_ky）—— 运行方法与说明

被测对象：项目根目录下的 `app.py`（Flask 情绪识别 Web 应用）
测试脚本：`test/test_ky/unitest_ky_white_1.py`
测试报告：`test/test_ky/ky_white_1_test_report.md`（每次运行自动重新生成）

---

## 1. 测试内容概览

测试脚本是一个可直接执行的 `unittest` 套件，包含 **3 个测试类、15 条用例（KY-TC-001 ~ KY-TC-015）**，覆盖：

| 测试类 | 用例 | 测试内容 |
| --- | --- | --- |
| `TestAppStartupFromProjectRoot` | KY-TC-001 ~ 003 | 真实子进程在项目根目录执行 `python app.py`，轮询 5000 端口并做 HTTP 探活；访问 `/` 与 `/upload` |
| `TestAppStartupFromParentDirectory` | KY-TC-004 | 在**上一级目录**以相对路径 `python emotion_detection_in_images/app.py` 启动，验证相对路径健壮性 |
| `TestAppWhiteBox` | KY-TC-005 ~ 015 | 进程内导入 `app` 模块，做白盒单元/集成测试 |

`TestAppWhiteBox` 具体覆盖：

- **常量与模块级对象**：`class_labels` 取值顺序（KY-TC-005）、`face_cascade` 加载可用（KY-TC-006）
- **`detect_faces_and_emotions` 分支覆盖**：
  - 无人脸早返回分支，且不修改原图（KY-TC-007）
  - 有人脸分支 + `argmax → class_labels` 标签映射 + 预处理规格 `(1,96,96,3)` 且归一化到 `[0,1]`（KY-TC-008，对 Haar 检测器与 `model.predict` 打桩）
- **路由注册表**：6 个业务路由及其 HTTP 方法（KY-TC-009）
- **`/upload` POST 分支**：multipart 上传无人脸图片，端到端返回 `No face detected`（KY-TC-010）
- **`/real_time` 与 `/video_feed`**：页面渲染、流式响应 `Content-Type`（KY-TC-011、012）
- **`/start` 与 `/stop` 判定覆盖**：重复点击不重复占用设备（KY-TC-013）、运行中停止释放设备并 302 重定向（KY-TC-014）、未启动即停止的异常安全性与幂等性（KY-TC-015）

测试手段：真实子进程命令行启动 + 端口/HTTP 探活 + Flask `test_client` + `unittest.mock` 打桩 + `template_rendered` 信号直接断言模板变量。

---

## 2. 环境要求

- **conda 环境 `emotion`**（TensorFlow + Flask + OpenCV + NumPy + h5py）
- 运行前需确保：
  1. `Models/model.h5` 与 `templates/` 目录存在（否则 `app.py` 无法导入/启动）
  2. **5000 端口空闲**（KY-TC-001 / KY-TC-004 的预置条件；端口被占用时这两条用例会被 `skip`）
  3. 无摄像头环境也可运行（`/video_feed` 用例只读响应头，不实际取流）

脚本不依赖当前工作目录，可在任意目录执行；内部会自行推导 `PROJECT_ROOT` 与上一级目录。

---

## 3. 运行方法

### 方式一：直接用 `emotion` 环境的解释器（推荐）

```bash
cd /home/ykan/emotion_detection_in_images
/home/ykan/miniconda3/envs/emotion/bin/python test/test_ky/unitest_ky_white_1.py
```

### 方式二：先激活环境再运行

```bash
conda activate emotion
cd /home/ykan/emotion_detection_in_images
python test/test_ky/unitest_ky_white_1.py
```

### 方式三：指定启动被测程序的解释器

脚本用 `KY_TEST_PYTHON` 环境变量决定**以哪个解释器启动被测的 `app.py` 子进程**；
未设置时默认使用 `sys.executable`（即运行本测试脚本的解释器）。

```bash
KY_TEST_PYTHON=/home/ykan/miniconda3/envs/emotion/bin/python \
  python test/test_ky/unitest_ky_white_1.py
```

> 注意：本脚本是带 `main()` 的独立运行器（负责采集控制台原文并渲染用例表），
> **不要**用 `python -m unittest` 或 `pytest` 直接加载，否则不会生成报告。

---

## 4. 运行时长与退出码

- **耗时**：脚本需真实加载 TensorFlow 模型并启动 Flask 服务，单次运行通常 **1~3 分钟**；
  节点超时上限见脚本常量 `STARTUP_TIMEOUT = 240` 秒。
- **退出码**：全部通过返回 `0`，存在失败/错误返回 `1`（可用于 CI 判定）。

---

## 5. 输出产物

运行结束后会生成/更新：

| 产物 | 说明 |
| --- | --- |
| 终端输出 | 用例执行过程（`verbosity=2`）+ 《软件测试用例表》 |
| `test/test_ky/ky_white_1_test_report.md` | 完整测试报告，每次运行**覆盖重写** |
| `test/test_ky/console_output/KY-TC-001_console.txt` | 根目录启动被测程序的 **stdout/stderr 合并原文**，不截断、不过滤 |
| `test/test_ky/console_output/KY-TC-004_console.txt` | 上一级目录启动被测程序的控制台原文 |

报告中的 Result 列引用控制台输出原文，仅将换行编码为 `<br>`、竖线转义为 `\|` 以适配 Markdown 表格，不做内容改写。

---

## 6. 结果解读与已知情况

### KY-TC-004（不通过，属预期暴露的缺陷）

在上一级目录以相对路径启动时，`app.py` 第 15 行 `load_model('Models/model.h5')` 使用相对路径，
工作目录非项目根目录时抛出 `FileNotFoundError`，进程 `returncode=1`。
该用例正是为暴露此相对路径健壮性问题而设计，**不属于测试脚本缺陷**。

其余 14 条用例在当前环境（无 CUDA 设备、无摄像头）下通过。
TensorFlow 启动日志中的 cuFFT/cuDNN/cuBLAS 注册告警与 `cuInit` 失败信息属正常现象，
TensorFlow 会自动回退到 CPU 执行，不影响测试结论。

### 用例被跳过的情况

- 5000 端口在启动前已被其他进程占用 → KY-TC-001 / KY-TC-004 标记为「未执行(预置条件不满足)」
- 依赖 KY-TC-001 的服务未就绪 → KY-TC-002 / KY-TC-003 跳过

---

## 7. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 报告未生成 / 只有终端输出 | 使用了 `python -m unittest` 或 `pytest`；请按第 3 节方式直接执行脚本 |
| 报错找不到 `Models/model.h5` | 当前环境缺少模型文件，或被测子进程的解释器不对；请确认 `Models/model.h5` 存在并设置 `KY_TEST_PYTHON` |
| KY-TC-001 被跳过 | 5000 端口被占用；先停掉占用进程（如上一次残留的 `app.py`）再重跑 |
| 运行结束但端口仍被占用 | 脚本以独立进程组启动被测程序，并在 `tearDownClass` 中用 `SIGTERM`（必要时 `SIGKILL`）回收整组进程 |
| `/video_feed` 用例异常 | 该用例只读响应头即结束；如出现异常请检查 `gen_frames` 是否在无摄像头环境下提前抛错 |
