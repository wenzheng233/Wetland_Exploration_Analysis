# Wetland Exploration & Analysis

面向沉湖湿地无人机影像的植物识别与分析项目。项目使用 Ultralytics YOLO 实例分割模型，包含视频抽帧、LabelMe 多边形标注转 YOLO 分割标签、数据集检查与划分、模型训练和推理，以及覆盖统计与可视化脚本。

> 仓库提供代码、配置和部分训练结果示例，不包含完整原始影像/标注数据集或模型权重。运行前需自行准备数据；训练时 Ultralytics 会按需下载预训练权重。

## 功能概览

- 将无人机视频按时间间隔抽帧，生成待标注图像。
- 将 LabelMe JSON 多边形标注转换为 YOLO 分割格式。
- 检查标注、分析类别和目标尺寸分布，并划分训练集与验证集。
- 基于 YOLO11s-seg 训练湿地植物实例分割模型，默认使用 1024 像素输入。
- 使用训练权重对图像或视频推理，并生成覆盖统计、结构可视化结果。

当前训练配置包含以下六类，类别 ID 顺序不可随意调整：

| ID | 类别 |
| --- | --- |
| 0 | Phragmites australis |
| 1 | Miscanthus sacchariflorus |
| 2 | Typha orientalis |
| 3 | Nelumbo nucifera |
| 4 | Alternanthera philoxeroides |
| 5 | Carex spp. |

## 项目结构

```text
configs/
  chenhu_seg.yaml          # YOLO 分割数据集与训练配置
  args.yaml                # 实验参数记录（含原开发环境路径）
data/
  selfdataset.txt          # 数据集说明/记录
docs/
  wetland_project_doc.docx # 项目文档
runs/segment/              # 已提交的训练曲线、混淆矩阵及预测示例
src/
  analysis/                # 推理、覆盖统计、结构可视化、地理标记处理
  evaluation/              # 验证脚本
  label/                   # 标注环境初始化与检查
  preprocessing/           # 抽帧、格式转换、数据分析和增强
  train/                   # 数据划分与模型训练
requirements.txt
```

## 环境安装

建议使用 Python 3.10 或 3.11，并根据本机 CUDA 版本先安装匹配的 PyTorch 与 torchvision。`requirements.txt` 当前锁定了 CUDA 12.9 构建的 PyTorch 2.8.0；如果本机不支持该构建，请先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装合适版本，再安装其余依赖。

```bash
git clone https://github.com/wenzheng233/Wetland_Exploration_n_Analysis-master.git
cd Wetland_Exploration_n_Analysis-master
python -m venv .venv
```

激活虚拟环境后安装依赖：

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux 使用：source .venv/bin/activate
python -m pip install -r requirements.txt
```

无 NVIDIA GPU 时，需将 `requirements.txt` 中的 `torch` 和 `torchvision` 版本替换为适合当前平台的构建。LabelImg/PyQt5 主要用于人工标注，不参与模型训练本身。

## 数据准备

### 1. 视频抽帧（可选）

将视频放到 `src/preprocessing/videos/`，运行：

```bash
python src/preprocessing/videos2images.py
```

脚本按默认 3 秒间隔抽帧，图像写入 `src/preprocessing/images/`。抽帧间隔可在脚本顶部修改。

### 2. 标注与格式转换

使用 LabelMe 对图像绘制多边形，JSON 中的 `label` 必须与 `src/preprocessing/json2txtyolo.py` 中的 `CLASSES` 完全一致。编辑该脚本的 `JSON_FOLDER`、`IMAGE_FOLDER` 后运行：

```bash
python src/preprocessing/json2txtyolo.py
```

转换结果默认写入 `output_json2txt/`。脚本只输出包含至少一个有效目标的图像和标签；没有有效目标、缺少图像或类别不匹配的标注会被跳过。

### 3. 划分数据集

`src/train/split_dataset.py` 默认从 `output_json2txt/output_images` 和 `output_json2txt/yolotxt` 读取数据，以固定随机种子 42 按 8:2 划分至 `datasets/chenhu_seg/`。如目录不同，先修改脚本中的 `SOURCE_ROOT` 和 `TARGET_ROOT`。

YOLO 分割数据集应符合如下结构：

```text
datasets/chenhu_seg/
  train/images/   train/labels/
  val/images/     val/labels/
```

图片与标签需同名（扩展名分别为图像格式和 `.txt`）。检查 `configs/chenhu_seg.yaml` 中 `path` 指向该数据集根目录；当前默认值 `./datasets/chenhu_seg` 相对于运行命令时的工作目录解析。运行标注初始化或数据分析脚本前，也需检查其文件顶部配置的输入路径。

## 训练

在仓库根目录运行：

```bash
python src/train/train_model.py
```

训练脚本使用 `yolo11s-seg.pt` 初始化，默认参数为 150 epochs、batch 8、imgsz 1024、SGD、GPU 0、workers 2。训练结果由 Ultralytics 保存到 `Chenhu_HighRes_Stable/s_batch8_worker2_1024/`（具体位置受 Ultralytics 默认 `runs/segment` 规则影响），权重位于输出目录的 `weights/best.pt` 和 `weights/last.pt`。显存不足时可降低 `batch` 或 `imgsz`；无 GPU 时需在脚本中将 `device=0` 改为 `device='cpu'`。

也可以直接通过 Ultralytics CLI 训练：

```bash
yolo segment train model=yolo11s-seg.pt data=configs/chenhu_seg.yaml imgsz=1024 batch=8 epochs=150
```

## 推理与分析

仓库中的分析脚本仍含有示例数据/权重路径。运行前请将路径改为本地真实存在的文件或目录：

- `src/analysis/predict.py`：单张图像推理并保存 `result.jpg`；调用时提供训练得到的权重路径。
- `src/evaluation/validation.py`：对 `test_videos/` 中的图像/视频执行预测；调整权重、输入目录和设备参数。
- `src/analysis/coverage_statistic.py`：统计每张图的检测框面积及比例，输出 CSV、饼图和箱线图。当前统计以检测框面积近似覆盖率，重叠框可能使总比例超过 100%。
- `src/analysis/structure_visualization.py`：将检测框绘制到输入图像并保存结构示意图；当前脚本最多处理前 20 张 JPG。
- `src/analysis/videos2geotagged_images.py`：视频/地理标记图像相关处理脚本，使用前请检查输入和输出路径及其所需元数据。

训练配置、类别名和 `src/preprocessing/json2txtyolo.py` 中的类别顺序必须保持一致。代码里的默认示例权重路径（例如 `runs/train/wetland_yolo11x_exp1/weights/best.pt`）并不对应仓库提交的训练示例目录，应替换为实际训练输出中的 `best.pt`。

## 依赖与结果

核心依赖及版本见 [`requirements.txt`](requirements.txt)。`runs/segment/Chenhu_HighRes_Stable/s_batch8_worker2_1024/` 中保留了部分训练图表与预测样例，便于查看实验结果；原始数据集、完整训练权重和大型中间文件不在仓库中。

## 注意事项

- 大型影像、数据集和模型权重应单独管理，不建议直接提交到 Git。
- `configs/args.yaml` 中的 `save_dir` 是原开发环境路径，跨机器使用时应更新或忽略该字段。
- 部分脚本用相对路径或固定目录名，建议从仓库根目录运行，并先检查脚本开头的路径配置。
- 当前仓库以实验脚本为主，个别分析脚本的路径、统计定义和类别颜色仍需根据实际数据核验。

## License

仓库当前未包含许可证文件。使用或再分发前，请联系项目维护者确认许可与数据集使用条件。
