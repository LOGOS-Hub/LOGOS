# LOGOS：科学生成对象的语言

<p align="center">
  <img src="logo/logos.png" alt="LOGOS" height="100">
</p>

<p align="center">
<a href="https://arxiv.org/abs/2510.24701" target="_blank"><img src="https://img.shields.io/badge/Technical Report-b5212f.svg?logo=arxiv" height="21px"></a>&nbsp;
<a href="https://github.com/placeholder/LOGOS"><img src="https://img.shields.io/badge/GitHub-LOGOS-181717?logo=github&logoColor=white" height="21px"></a>&nbsp;
<a href="README.md">English</a>
</p>

<p align="center">
<a href="https://huggingface.co/placeholder/LOGOS-8B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-LOGOS--8B-yellow" height="21px"></a>&nbsp;
<a href="https://huggingface.co/placeholder/LOGOS-pretrain-1B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-LOGOS--pretrain--1B-yellow" height="21px"></a>&nbsp;
<a href="https://huggingface.co/placeholder/LOGOS-pretrain-3B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-LOGOS--pretrain--3B-yellow" height="21px"></a>&nbsp;
<a href="https://huggingface.co/placeholder/LOGOS-pretrain-8B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-LOGOS--pretrain--8B-yellow" height="21px"></a>
</p>

## 概述

**LOGOS**（**L**anguage **O**f **G**enerative **O**bjects in **S**cience）是首个基于统一**科学语法**构建的多领域生成式框架。它将各种科学对象——蛋白质、抗体、小分子、化学反应、材料及其空间相互作用——编码为共享词表上的 token 序列，从而使单个自回归模型能够在自然科学各领域统一执行生成、预测与设计任务。

与依赖自然语言作为中介或需要显式三维几何网络的方法不同，LOGOS 直接基于领域原生表示进行建模。关键的空间关系（例如蛋白口袋–配体接触）被离散化并 token 化编码到统一语法中，使模型能够以纯序列的方式学习复杂的结构相互作用。

<p align="center">
  <img src="pics/LOGOS-mainfigure.png" alt="LOGOS Framework Overview" width="90%">
</p>

### 核心特性

* **统一科学语法**：将异构科学对象及其跨对象关系编码到统一离散 token 空间中的共享表征接口。
* **一模型多任务**：单个自回归模型即可处理蛋白质、小分子、材料、反应、抗体及其相互作用等多类任务。
* **无需显式三维几何**：通过 token 化表示捕捉空间接触与约束模式，不依赖几何神经网络或显式坐标。
* **预训练与下游对齐**：统一语法空间确保持续预训练目标与下游任务目标在形式上的一致性。

<p align="center">
  <img src="pics/logos-data-process.png" alt="Data Construction in LOGOS" width="90%">
</p>

## 仓库内容

本仓库提供 LOGOS 四个代表性下游任务的**推理脚本**：

| 任务 | 领域 | 脚本 | 说明 |
| ---- | ---- | ---- | ---- |
| 逆合成预测 | 化学 | `reversereact_gen.py` | 根据产物预测反应物 |
| 蛋白口袋识别 | 结构生物学 | `pocket_gen.py` | 从蛋白序列识别结合口袋 |
| 交互感知配体设计 | 药物发现 | `protein_ligand_interaction.py` | 生成可特异性结合蛋白口袋的配体 |
| 无条件材料生成 | 材料科学 | `material_generation.py` | 生成新颖且有效的材料 |

<p align="center">
  <img src="pics/bench_comparison.png" alt="Benchmark Comparison" width="90%">
</p>

## 环境配置

推荐使用英伟达官方 PyTorch Docker 镜像作为基础运行环境。

### 1. 拉取 NVIDIA PyTorch Docker 镜像

```bash
docker pull nvcr.io/nvidia/pytorch:25.02-py3
```

### 2. 启动容器

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.02-py3 bash
```

**环境要求：**
- NVIDIA Docker 镜像：`nvcr.io/nvidia/pytorch:25.02-py3`
- 支持 CUDA 的 GPU
- LOGOS 模型权重（请从 Hugging Face 下载）

## 快速开始

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("placeholder/LOGOS-8B")
tokenizer = AutoTokenizer.from_pretrained("placeholder/LOGOS-8B")

input_text = "<your_scientific_grammar_input>"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 各任务推理

### 1. 逆合成预测

根据产物分子生成反应物 SMILES。

```bash
python reversereact_gen.py \
    --data_file data/reversereact.jsonl \
    --model_path /path/to/checkpoint \
    --results_dir ./results \
    --num_samples 32 \
    --temperature 1.2 \
    --top_p 0.85 \
    --repetition_penalty 1.05 \
    --model_type llama
```

**输入 JSONL 格式：**
```json
{"test_input": "<MoleculeS>COCC)OC.....)cn1<MoleculeE><ReverseReact>", "groundtruth": "<MoleculeS>...<MoleculeE>"}
```

### 2. 蛋白口袋识别

根据输入 prompt 生成蛋白口袋序列。

```bash
python pocket_gen.py \
    --data_file data/pocket_trans.jsonl \
    --model_path /path/to/checkpoint \
    --results_dir ./results \
    --num_samples 40 \
    --batch_size 16 \
    --temperature 1.2 \
    --top_p 0.85 \
    --repetition_penalty 1.05 \
    --model_type llama
```

**输入 JSONL 格式：**
```json
{"id": "sample_001.pdb", "text": "<ProteinS>...<ProteinE><Search>"}
```

### 3. 交互感知配体设计

根据蛋白口袋 prompt 生成具有特异性结合能力的配体序列。

```bash
python protein_ligand_interaction.py \
    --data_file data/protein_ligand.jsonl \
    --model_path /path/to/checkpoint \
    --results_dir ./results \
    --num_samples 10 \
    --batch_size 16 \
    --temperature 1.2 \
    --top_p 0.85 \
    --repetition_penalty 1.05 \
    --model_type llama
```

**输入 JSONL 格式：**
```json
{"id": "sample_001.pdb", "protein_pocket_ligand": "<ProteinS>...", "groundtruth": "..."}
```

### 4. 无条件材料生成

以 `<MaterialS>` 为起始符生成材料结构序列，**无需输入数据文件**。

```bash
python material_generation.py \
    --model_path /path/to/checkpoint \
    --results_dir ./results \
    --num_samples 10000 \
    --batch_size 32 \
    --temperature 1.2 \
    --top_p 0.85 \
    --model_type llama
```

## 通用参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | （必填） | 模型权重路径 |
| `--results_dir` | str | `./results` | 输出结果目录 |
| `--top_p` | float | 因任务而异 | Nucleus 采样 top-p |
| `--temperature` | float | 因任务而异 | 采样温度 |
| `--repetition_penalty` | float | 1.05 | 重复惩罚系数 |
| `--num_samples` | int | 因任务而异 | 每个输入的生成次数 |
| `--batch_size` | int | 16 | 批处理大小 |
| `--model_type` | str | `llama` | 模型类型：`llama` 或 `qwen`（1B / 3B 模型选 `llama`，8B 模型选 `qwen`） |

## 输出格式

结果以 JSONL 文件保存在 `results_dir/<任务名>_<t>_<p>_<rp>/results.jsonl`。

**输出示例（逆合成）：**
```json
{"sequence": "...", "ppl": 12.34, "groundtruth": "..."}
```

**输出示例（材料生成）：**
```json
{"text": "<MaterialS>...<MaterialE>", "ppl": 8.56}
```

## 模型架构

LOGOS 基于自回归 Transformer 架构，并在统一科学语法上进行多领域持续预训练。模型参数规模覆盖 **1B 至 8B**，在该区间内观察到稳定的扩展行为。


## 许可证

本项目基于 **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)** 协议发布。

欢迎合作、反馈与社区贡献，共同推动面向自然科学的统一生成建模发展。
