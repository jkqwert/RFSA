# RFSA：汉画像石图文检索项目详细使用说明

## 📖 项目简介

本项目实现了一种基于**关系细粒度语义对齐（Relational Fine-Grained Semantic Alignment, RFSA）**的汉画像石图文检索方法。该方法通过将图像和文本分解为**四元组（主体subject、客体object、次要对象second、关系relation）**，实现细粒度的跨模态对齐，显著提升检索准确性。

### 核心特点

- **四元组分解**：将完整图像和文本拆分为主体、客体、次要对象、关系四个语义组件
- **双独立映射模块**：为图像和文本特征提供专用映射通道，映射到共享特征空间
- **自适应权重融合**：动态调整各组件特征的融合权重，提升检索稳定性和泛化能力
- **对比学习优化**：基于融合特征计算相似度，通过对比损失优化模型

---

## ⚠️ 首次使用必读：创建必要的文件夹和准备数据

由于数据集和模型文件较大，未包含在代码仓库中。**使用前请按以下步骤创建必要的文件夹并准备数据：**

### 1. 创建文件夹结构

在项目根目录下执行以下命令创建所需文件夹：

**Windows (PowerShell):**
```powershell
# 创建四元组图像目录
New-Item -ItemType Directory -Force -Path "com/subject"
New-Item -ItemType Directory -Force -Path "com/object"
New-Item -ItemType Directory -Force -Path "com/second object"
New-Item -ItemType Directory -Force -Path "com/relation"

# 创建特征存储目录
New-Item -ItemType Directory -Force -Path "features"

# 创建鲁棒性实验相关目录（可选）
New-Item -ItemType Directory -Force -Path "comblur/level1"
New-Item -ItemType Directory -Force -Path "comblur/level2"
New-Item -ItemType Directory -Force -Path "comblur/level3"
New-Item -ItemType Directory -Force -Path "featureslevel1"
New-Item -ItemType Directory -Force -Path "featureslevel2"
New-Item -ItemType Directory -Force -Path "featureslevel3"

# 创建输出目录
New-Item -ItemType Directory -Force -Path "outputs/mapping"
New-Item -ItemType Directory -Force -Path "outputs/shared_mapping"
New-Item -ItemType Directory -Force -Path "outputs/ablation"
New-Item -ItemType Directory -Force -Path "outputs/eval"
New-Item -ItemType Directory -Force -Path "outputs/shared_eval"
New-Item -ItemType Directory -Force -Path "outputs/swap_test"
New-Item -ItemType Directory -Force -Path "outputs/text_aug"
New-Item -ItemType Directory -Force -Path "outputs/level1"
New-Item -ItemType Directory -Force -Path "outputs/level2"
New-Item -ItemType Directory -Force -Path "outputs/level3"
New-Item -ItemType Directory -Force -Path "outputs/visualization"

# 创建训练/验证/测试图像目录（可选，根据需要）
New-Item -ItemType Directory -Force -Path "train_images"
New-Item -ItemType Directory -Force -Path "valid_images"
New-Item -ItemType Directory -Force -Path "test_images"
```

**Linux/macOS:**
```bash
# 创建四元组图像目录
mkdir -p com/subject com/object "com/second object" com/relation

# 创建特征存储目录
mkdir -p features

# 创建鲁棒性实验相关目录（可选）
mkdir -p comblur/level1 comblur/level2 comblur/level3
mkdir -p featureslevel1 featureslevel2 featureslevel3

# 创建输出目录
mkdir -p outputs/{mapping,shared_mapping,ablation,eval,shared_eval,swap_test,text_aug,level1,level2,level3,visualization}

# 创建训练/验证/测试图像目录（可选，根据需要）
mkdir -p train_images valid_images test_images
```

### 2. 需要准备的数据文件

| 文件/文件夹 | 说明 | 必需 |
|------------|------|------|
| `com/subject/` | 主体图像，命名格式：`{text_id}.png` | ✅ 是 |
| `com/object/` | 客体图像，命名格式：`{text_id}.png` | ✅ 是 |
| `com/second object/` | 次要对象图像，命名格式：`{text_id}.png` | ✅ 是 |
| `com/relation/` | 关系图像，命名格式：`{text_id}.png` | ✅ 是 |
| `clip_cn_vit-b-16.pt` | Chinese-CLIP预训练模型（ViT-B/16版本） | ✅ 是 |
| `clip_cn_rn50.pt` | Chinese-CLIP预训练模型（ResNet50版本） | ❌ 可选 |
| `train_images/` | 训练集原始完整图像 | ❌ 可选 |
| `valid_images/` | 验证集原始完整图像 | ❌ 可选 |
| `test_images/` | 测试集原始完整图像 | ❌ 可选 |

### 3. 预训练模型下载

Chinese-CLIP预训练模型可从以下地址获取：
- **官方GitHub**: https://github.com/OFA-Sys/Chinese-CLIP
- 下载 `clip_cn_vit-b-16.pt` 并放置在项目根目录

### 4. 数据准备检查清单

在开始训练前，请确保：
- [ ] `com/` 目录下的四个子文件夹都包含对应的图像文件
- [ ] `create.jsonl` 包含所有图像的四元组标注
- [ ] `train_texts.jsonl`、`valid_texts.jsonl`、`test_texts.jsonl` 包含正确的文本标注
- [ ] `clip_cn_vit-b-16.pt` 预训练模型文件存在
- [ ] `features/` 目录已创建（特征提取后会自动填充）

---

## 📁 项目结构

```
rfsa/
├── cn_clip/                      # Chinese-CLIP模型核心代码
│   ├── clip/                     # CLIP模型定义
│   ├── eval/                     # 评估脚本
│   ├── modeling/                 # 模型组件
│   └── training/                 # 训练相关
│
├── com/                          # ⚠️ 需创建 - 原始四元组图像目录
│   ├── subject/                  # 主体图像
│   ├── object/                   # 客体图像
│   ├── second object/            # 次要对象图像
│   └── relation/                 # 关系图像
│
├── comblur/                      # ⚠️ 需创建（可选）- 扰动后的图像（鲁棒性实验）
│   ├── level1/                   # 轻度扰动
│   ├── level2/                   # 中度扰动
│   └── level3/                   # 重度扰动
│
├── features/                     # ⚠️ 需创建 - 提取的特征文件（运行特征提取后自动生成）
│   ├── subject_features.json     # 主体图像特征
│   ├── object_features.json      # 客体图像特征
│   ├── second_object_features.json  # 次要对象图像特征
│   ├── relation_features.json    # 关系图像特征
│   ├── subject_text_features.json   # 主体文本特征
│   ├── object_text_features.json    # 客体文本特征
│   ├── second_text_features.json    # 次要对象文本特征
│   └── relation_text_features.json  # 关系文本特征
│
├── featureslevel1/               # ⚠️ 需创建（可选）- Level1扰动特征
├── featureslevel2/               # ⚠️ 需创建（可选）- Level2扰动特征
├── featureslevel3/               # ⚠️ 需创建（可选）- Level3扰动特征
│
├── outputs/                      # ⚠️ 需创建 - 输出目录
│   ├── mapping/                  # 映射模型训练输出
│   ├── shared_mapping/           # 共享映射模型输出
│   ├── ablation/                 # 消融实验输出
│   ├── eval/                     # 评估结果
│   ├── shared_eval/              # 共享模型评估结果
│   ├── swap_test/                # 主客体交换实验
│   ├── text_aug/                 # 文本扰动数据
│   ├── level1/                   # Level1鲁棒性评估
│   ├── level2/                   # Level2鲁棒性评估
│   ├── level3/                   # Level3鲁棒性评估
│   └── visualization/            # 可视化结果
│
├── train_images/                 # ⚠️ 需创建（可选）- 训练集图像
├── valid_images/                 # ⚠️ 需创建（可选）- 验证集图像
├── test_images/                  # ⚠️ 需创建（可选）- 测试集图像
│
├── train_texts.jsonl             # ✅ 已包含 - 训练集文本标注
├── valid_texts.jsonl             # ✅ 已包含 - 验证集文本标注
├── test_texts.jsonl              # ✅ 已包含 - 测试集文本标注
├── create.jsonl                  # ✅ 已包含 - 四元组标注文件
│
├── clip_cn_vit-b-16.pt           # ⚠️ 需下载 - Chinese-CLIP预训练模型（ViT-B/16）
├── clip_cn_rn50.pt               # ⚠️ 需下载（可选）- Chinese-CLIP预训练模型（ResNet50）
│
├── extract_composite_features.py # 特征提取脚本
├── train_mapping.py              # 映射模型训练脚本
├── train_shared_mapping.py       # 共享映射模型训练脚本
├── train_ablation.py             # 消融实验训练脚本
├── eval_mapping.py               # 映射模型评估脚本
├── eval_shared_mapping.py        # 共享映射模型评估脚本
├── eval_ablation.py              # 消融实验评估脚本
├── mapping_model.py              # 映射模型定义
├── mapping_model_shared.py       # 共享映射模型定义
├── mapping_model_ablation.py     # 消融实验模型定义
├── data_loader.py                # 数据加载器
├── data_loader_ablation.py       # 消融实验数据加载器
├── feature_loader_ablation.py    # 消融实验特征加载器
├── augment_perturbations.py      # 数据扰动脚本
├── test_subject_object_swap.py   # 主客体交换实验
├── compare_alignments.py         # 对齐效果对比
├── visualize_alignment.py        # 对齐可视化
├── visualization_utils.py        # 可视化工具
├── plot_relationship_validity.py # 关系有效性绘图
├── alignment_metrics.py          # 对齐指标计算
├── compute_similarity.py         # 相似度计算
├── eval_utils.py                 # 评估工具函数
├── image_augmentation.py         # 图像增强
├── comblur.py                    # 模糊处理
├── trans.py                      # 转换工具
├── fix_jsonl.py                  # JSONL修复工具
└── check_duplicates.py           # 重复检查工具
```

> 📌 **说明**：标记为 `⚠️ 需创建` 的文件夹需要用户手动创建，标记为 `⚠️ 需下载` 的文件需要用户自行下载。

---

## 🔧 环境配置

### 1. 依赖安装

```bash
# 创建虚拟环境（推荐）
conda create -n rfsa python=3.8
conda activate rfsa

# 安装PyTorch（根据CUDA版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy scikit-learn tqdm tensorboard pillow
pip install transformers
```

### 2. 预训练模型下载

项目需要Chinese-CLIP预训练模型：

| 模型文件 | 说明 | 下载地址 |
|---------|------|---------|
| `clip_cn_vit-b-16.pt` | ViT-B/16版本（**推荐**） | [Chinese-CLIP GitHub](https://github.com/OFA-Sys/Chinese-CLIP) |
| `clip_cn_rn50.pt` | ResNet50版本（可选） | [Chinese-CLIP GitHub](https://github.com/OFA-Sys/Chinese-CLIP) |

下载后将模型文件放置在项目根目录。

---

## 📊 数据格式说明

### 1. 文本标注文件格式（train/valid/test_texts.jsonl）

每行为一个JSON对象，格式如下：
```json
{"text_id": "000308", "text": "图中一条龙和一只猛兽在围攻一人。", "image_ids": ["000600"]}
```

字段说明：
- `text_id`: 文本唯一标识符
- `text`: 完整的文本描述
- `image_ids`: 对应的图像ID列表（一个文本可对应多个图像）

### 2. 四元组标注文件格式（create.jsonl）

每行为一个JSON对象，格式如下：
```json
{"text_id": "000021", "subject": "两只猛兽", "object": "一条龙", "second": "", "relation": "两只猛兽在围攻一条龙"}
```

字段说明：
- `text_id`: 与texts.jsonl中的text_id对应
- `subject`: 主体（动作发出者）
- `object`: 客体（动作接受者）
- `second`: 次要对象（场景中的其他元素）
- `relation`: 关系描述（主客体之间的动作/关系）

### 3. 四元组图像目录结构

```
com/
├── subject/          # 主体图像，命名格式：{text_id}.png
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── object/           # 客体图像
├── second object/    # 次要对象图像
└── relation/         # 关系图像
```

---

## 🚀 使用流程

### 完整流程概览

```
1. 数据准备 → 2. 特征提取 → 3. 模型训练 → 4. 模型评估 → 5. 可视化分析
```

---

### Step 1: 特征提取

提取四元组图像和文本的特征：

```bash
python extract_composite_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --subject-dir com/subject \
    --object-dir com/object \
    --second-object-dir "com/second object" \
    --relation-dir com/relation \
    --text-data create.jsonl \
    --output-dir features \
    --resume clip_cn_vit-b-16.pt
```

**参数说明：**
| 参数 | 说明 |
|------|------|
| `--extract-image-feats` | 提取图像特征 |
| `--extract-text-feats` | 提取文本特征 |
| `--subject-dir` | 主体图像目录 |
| `--object-dir` | 客体图像目录 |
| `--second-object-dir` | 次要对象图像目录 |
| `--relation-dir` | 关系图像目录 |
| `--text-data` | 四元组标注文件 |
| `--output-dir` | 特征输出目录 |
| `--resume` | Chinese-CLIP预训练模型路径 |

**输出文件：**
- `features/subject_features.json` - 主体图像特征
- `features/object_features.json` - 客体图像特征
- `features/second_object_features.json` - 次要对象图像特征
- `features/relation_features.json` - 关系图像特征
- `features/subject_text_features.json` - 主体文本特征
- `features/object_text_features.json` - 客体文本特征
- `features/second_text_features.json` - 次要对象文本特征
- `features/relation_text_features.json` - 关系文本特征

---

### Step 2: 模型训练

#### 方式一：共享映射模型训练（推荐）

```bash
python train_shared_mapping.py \
    --train-texts train_texts.jsonl \
    --valid-texts valid_texts.jsonl \
    --create-jsonl create.jsonl \
    --text-features-dir features \
    --image-features-dir features \
    --output-dir outputs/shared_mapping \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --temperature 0.07
```

#### 方式二：标准映射模型训练

```bash
python train_mapping.py \
    --train-texts train_texts.jsonl \
    --valid-texts valid_texts.jsonl \
    --create-jsonl create.jsonl \
    --text-features-dir features \
    --image-features-dir features \
    --clip-checkpoint clip_cn_vit-b-16.pt \
    --output-dir outputs/mapping \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-4
```

**训练参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train-texts` | 训练集文本标注 | - |
| `--valid-texts` | 验证集文本标注 | - |
| `--create-jsonl` | 四元组标注文件 | - |
| `--text-features-dir` | 文本特征目录 | - |
| `--image-features-dir` | 图像特征目录 | - |
| `--output-dir` | 输出目录 | - |
| `--batch-size` | 批次大小 | 32 |
| `--epochs` | 训练轮数 | 50 |
| `--lr` | 学习率 | 1e-4 |
| `--temperature` | 对比学习温度参数 | 0.07 |

**输出文件：**
- `outputs/shared_mapping/best_model.pt` - 最优模型权重
- `outputs/shared_mapping/training.log` - 训练日志
- `outputs/shared_mapping/events.out.tfevents.*` - TensorBoard日志

---

### Step 3: 模型评估

#### 评估共享映射模型

```bash
python eval_shared_mapping.py \
    --test-texts test_texts.jsonl \
    --create-jsonl create.jsonl \
    --text-features-dir features \
    --image-features-dir features \
    --checkpoint outputs/shared_mapping/best_model.pt \
    --output-dir outputs/shared_eval \
    --batch-size 32
```

#### 评估标准映射模型

```bash
python eval_mapping.py \
    --test-texts test_texts.jsonl \
    --create-jsonl create.jsonl \
    --text-features-dir features \
    --image-features-dir features \
    --checkpoint outputs/mapping/best_model.pt \
    --output-dir outputs/eval \
    --batch-size 32
```

**评估输出：**
- `Recall@1`, `Recall@5`, `Recall@10` - 召回率指标
- `MeanR` - 平均召回率
- `R@sum` - 召回率总和
- 双向检索结果（文本到图像、图像到文本）

---

### Step 4: 消融实验

消融实验用于验证各模块的有效性：

```bash
python eval_ablation.py \
    --test-texts test_texts.jsonl \
    --create-jsonl create.jsonl \
    --text-features-dir features \
    --image-features-dir features \
    --checkpoint outputs/ablation/111/best_model.pt \
    --output-dir outputs/ablation/111/eval \
    --batch-size 32 \
    --use-prompt \
    --use-component \
    --use-shared-space
```

**消融配置选项：**
| 参数 | 说明 |
|------|------|
| `--use-prompt` | 启用提示学习模块 |
| `--use-component` | 启用组件级特征融合 |
| `--use-shared-space` | 启用共享特征空间映射 |

---

### Step 5: 关系有效性验证（主客体交换实验）

```bash
python test_subject_object_swap.py \
    --checkpoint outputs/shared_mapping/best_model.pt \
    --test-texts test_texts.jsonl \
    --create-jsonl create.jsonl \
    --text-features-dir features \
    --image-features-dir features \
    --output-dir outputs/swap_test \
    --batch-size 32
```

该实验通过交换主客体验证模型对细粒度关系语义的理解能力。

---

### Step 6: 鲁棒性实验

#### 6.1 生成扰动数据

```bash
python augment_perturbations.py \
    --image-root "D:\python project\sorclip\com" \
    --image-output-root "D:\python project\sorclip\comblur" \
    --text-json "create.jsonl" \
    --text-output-dir "outputs/text_aug" \
    --levels 1 2 3
```

**扰动级别说明：**
| 级别 | 图像扰动 | 文本扰动 |
|------|----------|----------|
| L1 | 噪声添加、亮度/对比度调整、轻微几何变换 | 同音字替换、同义词替换、标点变化 |
| L2 | 局部遮挡、几何形变、颜色空间扰动 | 关键词缺失、文本长度调整、词序调整 |
| L3 | 重度噪声和模糊、大面积遮挡 | 核心关键词替换、关键词乱序、文本片段缺失 |

#### 6.2 提取扰动数据特征

```bash
# 以Level 3为例
python extract_composite_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --subject-dir comblur/level3/subject \
    --object-dir comblur/level3/object \
    --second-object-dir "comblur/level3/second object" \
    --relation-dir comblur/level3/relation \
    --text-data outputs/text_aug/create_level3.jsonl \
    --output-dir featureslevel3 \
    --resume clip_cn_vit-b-16.pt
```

#### 6.3 评估扰动数据

```bash
python eval_shared_mapping.py \
    --test-texts test_texts.jsonl \
    --create-jsonl outputs/text_aug/create_level3.jsonl \
    --text-features-dir featureslevel3 \
    --image-features-dir featureslevel3 \
    --checkpoint outputs/shared_mapping/best_model.pt \
    --output-dir outputs/level3 \
    --batch-size 32
```

---

### Step 7: 可视化对比

```bash
python compare_alignments.py \
    --checkpoints outputs/ablation/100/best_model.pt outputs/shared_mapping/best_model.pt \
    --config-names "100" "111" \
    --test-texts test_texts.jsonl \
    --create-jsonl create.jsonl \
    --text-features-dir features \
    --image-features-dir features \
    --output-dir outputs/visualization/comparison \
    --batch-size 32
```

---

## 📈 评估指标说明

| 指标 | 说明 |
|------|------|
| **Recall@1 (R@1)** | Top-1召回率，正确结果在第1位的比例 |
| **Recall@5 (R@5)** | Top-5召回率，正确结果在前5位的比例 |
| **Recall@10 (R@10)** | Top-10召回率，正确结果在前10位的比例 |
| **MeanR** | 平均召回率 (R@1+R@5+R@10)/3 |
| **R@sum** | 召回率总和 R@1+R@5+R@10（两个方向） |

---

## 💡 常见问题

### Q1: CUDA内存不足

**解决方案：**
- 减小 `--batch-size` 参数
- 使用 `--fp16` 进行混合精度训练（如果支持）

### Q2: 特征文件找不到

**解决方案：**
- 确保已运行 `extract_composite_features.py` 完成特征提取
- 检查 `--text-features-dir` 和 `--image-features-dir` 路径是否正确

### Q3: 模型加载失败

**解决方案：**
- 确认 `clip_cn_vit-b-16.pt` 预训练模型文件存在
- 检查模型文件是否完整（未损坏）

### Q4: 四元组标注缺失

**解决方案：**
- 检查 `create.jsonl` 中是否包含所有 `text_id` 的四元组标注
- 确保四元组标注格式正确

### Q5: 找不到文件夹/目录不存在

**解决方案：**
- 请参考本文档开头的「首次使用必读」部分，创建所有必要的文件夹
- 使用提供的命令一键创建文件夹结构

### Q6: 缺少预训练模型文件

**解决方案：**
- 从 [Chinese-CLIP GitHub](https://github.com/OFA-Sys/Chinese-CLIP) 下载预训练模型
- 将 `clip_cn_vit-b-16.pt` 放置在项目根目录

---

## 📚 相关文档

| 文档 | 说明 |
|------|------|
| [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) | 模型架构详解 |
| [ABLATION_STUDY.md](ABLATION_STUDY.md) | 消融实验说明 |
| [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) | 可视化指南 |
| [INNOVATIONS.md](INNOVATIONS.md) | 创新点说明 |

---

## 🔗 快速命令参考

```bash
# 1. 特征提取
python extract_composite_features.py --extract-image-feats --extract-text-feats --subject-dir com/subject --object-dir com/object --second-object-dir "com/second object" --relation-dir com/relation --text-data create.jsonl --output-dir features --resume clip_cn_vit-b-16.pt

# 2. 模型训练
python train_shared_mapping.py --train-texts train_texts.jsonl --valid-texts valid_texts.jsonl --create-jsonl create.jsonl --text-features-dir features --image-features-dir features --output-dir outputs/shared_mapping --batch-size 32 --epochs 50 --lr 1e-4 --temperature 0.07

# 3. 模型评估
python eval_shared_mapping.py --test-texts test_texts.jsonl --create-jsonl create.jsonl --text-features-dir features --image-features-dir features --checkpoint outputs/shared_mapping/best_model.pt --output-dir outputs/shared_eval --batch-size 32

# 4. 主客体交换实验
python test_subject_object_swap.py --checkpoint outputs/shared_mapping/best_model.pt --test-texts test_texts.jsonl --create-jsonl create.jsonl --text-features-dir features --image-features-dir features --output-dir outputs/swap_test --batch-size 32
```

---

## 📧 联系方式

如有问题，请提交Issue或联系项目维护者。

2476270892@qq.com

**最后更新：2026年1月**