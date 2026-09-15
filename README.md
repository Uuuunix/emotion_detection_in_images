# 图像情绪检测系统测试作业

本项目为hust软件学院的软件测试课程项目

## 项目结构

```text
Models/                   # 训练后的模型权重
Notebooks/                # 数据探索、模型训练与评估笔记本
Screenshots/              # 应用界面截图
static/                   # CSS、图标等静态资源
templates/                # HTML 页面模板
test/                     # 白盒测试脚本，使用 unittest 和 pytest
app.py                    # Flask 应用入口
pytest.ini                # pytest 测试发现配置
requirements.txt          # 原始依赖快照
requirements-macos.txt    # macOS 运行依赖
requirements-test.txt     # 额外测试依赖
.gitattributes            # Git LFS 配置
.gitignore                # Git 忽略配置
```

图标来源：[Freepik](https://www.freepik.com/icon/smile_2383590)。

## 数据集与模型

项目使用 [FER-2013 数据集](https://www.kaggle.com/datasets/msambare/fer2013)，该数据集包含 35,887 张带标签的 48×48 像素灰度人脸图像。本项目选用其中的 `Happy`、`Sad`、`Surprise`、`Neutral` 四个类别。

模型训练采用以下方法：

- **数据增强**：通过旋转、翻转、缩放等变换增加训练样本的变化。
- **Dropout 与 L2 正则化**：降低过拟合风险。
- **批归一化**：稳定中间层输出，辅助训练。
- **EarlyStopping**：验证指标不再改善时停止训练。
- **ReduceLROnPlateau**：根据指标变化调整学习率。
- **ModelCheckpoint**：保存训练过程中表现较好的模型。

## 安装与启动

### 1. 克隆仓库

```bash
git clone https://github.com/TouradBaba/emotion_detection_in_images.git
cd emotion_detection_in_images
```

### 2. 创建虚拟环境

本项目已使用 Python 3.11 和 TensorFlow 2.18 在 macOS 环境运行。后续命令均应在项目根目录执行，因为应用通过相对路径加载 `Models/model.h5`。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows 可在命令提示符中使用以下命令：

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. 安装运行依赖

macOS 使用专用依赖文件：

```bash
python -m pip install -r requirements-macos.txt
```

`requirements.txt` 保留原始依赖快照，采用 UTF-16 编码，与这里使用的 macOS 依赖文件分开维护。以上依赖安装步骤对应已验证的 macOS 环境；Windows 命令示例仅说明虚拟环境和环境变量的设置方式。

### 4. 获取模型权重

模型通过 Git LFS 管理。安装 Git LFS 后执行：

```bash
git lfs install
git lfs pull
```

确认 `Models/model.h5` 是实际模型文件，大小约 195 MB，而非只有少量文本的 Git LFS 指针文件。

### 5. 启动 Flask 服务

现有 HDF5 模型需要使用旧版 Keras 兼容模式。macOS 依赖中已包含 `tf-keras`，启动时在导入 TensorFlow 前设置环境变量：

```bash
TF_USE_LEGACY_KERAS=1 python -m flask --app app run --host 127.0.0.1 --port 5055
```

Windows 命令提示符中的写法为：

```bat
set TF_USE_LEGACY_KERAS=1
python -m flask --app app run --host 127.0.0.1 --port 5055
```

### 6. 打开应用

浏览器访问 <http://127.0.0.1:5055>。本地测试使用 5055 端口以避开当时已被占用的端口；若该端口也被占用，可自行更换。

直接运行 `python app.py` 时，默认使用 5000 端口，同样需要提前设置 `TF_USE_LEGACY_KERAS=1`。

## 使用方法

### 图片上传

1. 从首页进入 **Upload** 页面。
2. 选择图片并提交。本地格式测试覆盖了灰度 JPEG、带透明边缘的 PNG、BMP 和 WEBP。
3. 查看返回图片中的人脸框、表情标签和页面顶部结果。

### 实时检测

1. 进入 **Real-Time Detection** 页面。
2. 点击 **Start** 启动摄像头。
3. 查看视频中的表情检测结果。
4. 点击 **Stop** 释放摄像头。

应用访问运行 Flask 的计算机上的 0 号摄像头。macOS 弹出权限提示时，需要允许运行 Python 的应用访问摄像头。`wl-bb-01` 导航测试未启动摄像头，也未验证实际视频采集。

## 测试说明

### 安装测试依赖

先安装运行依赖，再执行：

```bash
python -m pip install -r requirements-test.txt
```

### 运行测试

在项目根目录按需执行：

```bash
# WL：WB-01～WB-06，以及新增的多脸结果完整性失败用例
TF_USE_LEGACY_KERAS=1 python -m unittest test.unitest_wl -v

# JYH：WB-07～WB-18 及缺陷用例，覆盖上传、预处理和摄像头控制
TF_USE_LEGACY_KERAS=1 python -m pytest test/unitest_jyh.py -v

# KY：预处理、标签映射和应用启动等检查
TF_USE_LEGACY_KERAS=1 python test/test_ky/unitest_ky_white_1.py
```

Windows 下先按照启动说明设置环境变量，再执行对应的 `python` 命令，无需在命令前重复写内联环境变量。

`pytest.ini` 配置了 `unitest_*.py` 和 `test_*.py` 两种文件名的测试发现规则。上述命令用于分别选择测试套件；WL 的 unittest 命令运行类中的 7 条用例，不包含文件中额外的模块级 pytest 测试函数。

各套件的依赖与判定方式如下：

- **WL**：用测试替身替换模型加载和预测，保留真实图像处理操作。多脸完整性用例要求返回全部表情，目前会暴露仅返回最后一次预测的问题。
- **JYH**：收集测试时会导入应用，因此需要真实模型文件；单条用例会对部分操作打桩。D-01、D-02、D-03 缺陷用例使用严格 `xfail` 标记，预期失败不代表缺陷验收通过。
- **KY**：启动测试会创建子进程，需确保 5000 端口空闲，启动等待上限为 240 秒。可通过 `KY_TEST_PYTHON` 指定子进程使用的 Python 解释器。

### 测试用例汇总

以下统计依据《测试用例清单(3).xlsx》中 `Test Cases测试用例` 工作表第 2～35 行，于 2026 年 9 月 15 日核对。数据来自清单中已记录的执行结果，本次 README 更新未重新运行测试。

| 用例编号 | 测试范围 | 总数 | 通过 OK | 失败 NG | 部分通过 POK |
| --- | --- | ---: | ---: | ---: | ---: |
| `jyh-01`～`jyh-08` | 上传分支、首次启动摄像头、非法输入处理 | 8 | 6 | 2 | 0 |
| `wl-01`～`wl-07` | 人脸检测分支、预处理、标签映射、多脸结果完整性 | 7 | 6 | 1 | 0 |
| `ky-wb-1`～`ky-wb-11` | 启动路径、模块初始化、路由和摄像头状态 | 11 | 10 | 1 | 0 |
| `jyh-bb-01`～`jyh-bb-04` | 多人、无脸、非法文件和空提交场景 | 4 | 0 | 3 | 1 |
| `wl-bb-01`～`wl-bb-04` | 页面导航、上传页初始状态、四类表情和图片格式兼容性 | 4 | 3 | 0 | 1 |
| **合计** | **26 条白盒用例、8 条黑盒用例** | **34** | **25** | **7** | **2** |

严格通过率为 **25/34 = 73.53%**，失败率为 20.59%，部分通过率为 5.88%，无法执行（NT）占比为 0%。POK 和 pytest XFAIL 不计入验收通过数。

其中，`jyh-04` 和 `jyh-05` 的预期是复现当前 400/500 异常路径，因此记为 OK；`jyh-07` 和 `jyh-08` 要求对相应异常输入进行友好处理，因此记为 NG。验证现有行为通过，不代表该行为满足缺陷修复要求。

清单的基本信息页存在总数公式异常，显示了负数总量及相关比例。上表按 34 条有效明细及其状态重新统计，未采用异常汇总公式的结果。

### 自动化情况与脚本对应关系

清单中 19 条用例明确标记为自动化、4 条标记为非自动化，11 条 KY 用例的自动化字段为空。按已填写字段可确认的自动化比例为 **19/34 = 55.88%**。

KY 用例步骤包含脚本检查，但是否计入自动化用例仍需补全并确认相应字段。现有字段尚不足以确认已达到“至少 24 条自动化用例”的要求。

| 清单编号 | 脚本或证据入口 | 对应关系 |
| --- | --- | --- |
| `jyh-01`～`jyh-06` | `test/unitest_jyh.py` | `test_wb07_*`～`test_wb12_*` 函数 |
| `jyh-07`、`jyh-08` | `test/unitest_jyh.py` | D-02、D-01 验收用例，标记为严格 `xfail` |
| `wl-01`～`wl-07` | `test/unitest_wl.py` | `WhiteBoxTests` 中的 7 个方法；`wl-07` 为多脸失败用例 |
| `ky-wb-1`～`ky-wb-11` | `test/test_ky/unitest_ky_white_1.py` | 启动与应用检查；脚本中的 KY-TC 编号与清单编号不同 |
| `jyh-bb-01`～`jyh-bb-04` | 清单中的操作步骤 | 真实 HTTP 上传和页面结果检查，清单标记为非自动化 |
| `wl-bb-01`～`wl-bb-04` | `outputs/blackbox-01-04-20260911/` | 浏览器脚本检查与人工截图核验，清单标记为自动化；证据目录保存在本地 |

仓库还包含清单之外的测试，因此测试运行器发现的用例数量不一定等于表中的 34 条。

### 黑盒测试发现

- **`jyh-bb-01`：NG**。三人图中检测出 3 个人脸框，但页面顶部只显示 Happy，未完整展示三人的结果。
- **`jyh-bb-02`：NG**。纯色图返回 `No face detected`；风景图和猫图分别误报为 Happy 和 Sad。
- **`jyh-bb-03`：NG**。伪装成 JPEG 的文本、PDF 返回 HTTP 500；截断 JPEG 返回 `No face detected`，没有明确的无效图片提示。
- **`jyh-bb-04`：POK**。前端必填校验和空文件名重定向正常，但缺少 `image` 字段的请求返回默认 HTTP 400。
- **`wl-bb-01`：OK**。首页、上传页和实时检测页均返回 HTTP 200，入口、Home 导航和 favicon 正常；该用例未启动摄像头。
- **`wl-bb-02`：OK**。上传页初始状态返回 HTTP 200，未显示检测结果文字或结果图片。
- **`wl-bb-03`：POK**。四类样本均返回 HTTP 200 并显示人脸框和标签，但 Neutral 被识别为 Sad，仅 3/4 标签一致；所用低分辨率放大样本也未满足原高清图片条件。
- **`wl-bb-04`：OK**。透明 PNG、BMP、WEBP 和灰度 JPG 均返回 HTTP 200 及可解码的 JPEG 结果；同时观察到背景被误检为人脸。

KY 中与摄像头相关的检查覆盖响应头及受控摄像头状态，不能据此认定物理摄像头实时采集已经验证通过。

### 黑盒测试证据

本地 `outputs/blackbox-01-04-20260911/` 保存 `wl-bb-01`～`wl-bb-04` 的导航、初始状态、四类表情上传和格式兼容性测试证据，包括截图、JSON 结果和执行记录。这 4 条用例已经纳入上述 34 条统计；证据文件中的 BB-01～BB-04 分别对应清单中的 `wl-bb-01`～`wl-bb-04`。

`outputs/test-report-20260913/` 中的早期报告仅覆盖 11 条用例，尚未按本次汇总清单更新。这两个目录已被 Git 忽略，新克隆的仓库不包含这些本地材料。

原 6 条 WL 白盒用例对 `detect_faces_and_emotions` 达到了 100% 语句覆盖率和分支覆盖率。该覆盖率记录早于新增失败用例，不能代表清单中全部 34 条用例、整个 `app.py` 或端到端功能的覆盖情况。

### 已知问题

- **多脸结果丢失**：结果图中有多个标签，但函数返回值和页面顶部仅保留最后一次预测。对应 D-03、`wl-07`、`jyh-bb-01`。
- **非法图片触发 500**：解码失败的 `None` 进入检测函数，返回 HTTP 500，没有友好拒绝。对应 D-01、`jyh-08`、`jyh-bb-03`。
- **缺少上传字段时处理不友好**：请求没有 `image` 字段时返回 Flask 默认 HTTP 400。对应 D-02、`jyh-07`、`jyh-bb-04`。
- **启动依赖工作目录**：从上一级目录启动时，无法通过相对路径找到 `Models/model.h5`。对应 `ky-wb-2`；目前应在项目根目录按上述命令启动。
- **绘图污染模型输入**：函数先绘制边框，再裁剪人脸区域，导致标注像素进入模型输入。
- **人脸误检**：风景图、猫图产生误报，对应 `jyh-bb-02`；格式兼容性测试中也出现背景物体被当成人脸的情况，对应 `wl-bb-04`。

清单记录了上述未解决问题，尚未记录相应修复完成和回归验证通过的结果。

## 模型评估指标

原项目报告的整体分类准确率为 **79.9%**。以下保留原有评估数据，本轮功能测试未重新验证这些指标。

| 表情类别 | 精确率 Precision | 召回率 Recall | F1 分数 | 样本数 Support |
| --- | ---: | ---: | ---: | ---: |
| Happy（高兴） | 0.91 | 0.89 | 0.90 | 1774 |
| Sad（悲伤） | 0.69 | 0.76 | 0.72 | 1247 |
| Surprise（惊讶） | 0.86 | 0.86 | 0.86 | 831 |
| Neutral（中性） | 0.72 | 0.67 | 0.69 | 1233 |

- **精确率**：被预测为某类别的样本中，实际属于该类别的比例。
- **召回率**：实际属于某类别的样本中，被正确识别出的比例。
- **F1 分数**：精确率和召回率的调和平均数。
- **样本数**：评估集中对应类别的样本数量。

## 界面截图

首页、图片上传页和实时检测页如下：

![首页、图片上传页和实时检测页](Screenshots/Screenshots_of_the_Three_Pages.png)

## 代码风格

项目使用 PEP 8 风格规范，并通过类型注解说明输入输出类型，通过文档字符串说明函数用途。
