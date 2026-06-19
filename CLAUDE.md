# CLAUDE.md — 项目协作规范

## 项目概述
期末大作业：复现论文 "Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting"
截止日期：2026/6/21（周日）23:59

## 技术栈
- Python >= 3.11
- 环境管理：uv（不用 conda/pip）
- 深度学习：PyTorch（优先 torch.nn，避免 Lightning 等高封装库，方便理解细节）
- 数值计算：numpy, pandas
- 可视化：matplotlib, seaborn
- 统计/基线：statsmodels（SARIMA）
- 可解释性：shap（可选创新点）

## 项目结构
```
├── src/               # 核心源码
│   ├── models.py      # 模型定义（LSTM, GRU, 基线等）
│   ├── preprocessing.py
│   ├── train.py       # 训练循环
│   ├── evaluate.py    # 评估与指标
│   ├── config.py      # 配置（不硬编码）
│   └── utils.py       # 工具函数
├── notebooks/         # Jupyter 探索/分析
├── experiments/       # 实验输出（logs/checkpoints/results）
├── data/              # 数据集
├── report/            # 实验报告
├── pyproject.toml
├── CLAUDE.md          # 本文件
├── requirements.md    # 要求文档
└── plan.md            # 计划文档
```
## 详细结构要求
1. 不同模块代码分开不同的文件写，不要一个文件写过长
2. 关键参数等的修改写在脚本启动文件处，做到实验留痕

## 协作方式
1. **先确认，再动手**：每次开始新模块前，我先说明思路和代码结构，你确认后再写
2. **逐模块推进**：按 plan.md 的 Day 1→5 顺序，完成一个再下一个
3. **每步可运行**：每写完一个模块，确保能单独跑通再继续
4. **不懂就问**：我写的每段代码，你都可以问我"为什么这么写"，我会解释到你能理解为止
5. **代码风格**：
   - 变量/函数：snake_case
   - 类名：PascalCase
   - 核心函数加类型注解
   - 只写必要注释，用代码自解释
6. **留足边界感并给你的学习空间**：涉及到的核心代码修改，环境配置问题，论文阅读问题，务必先告知，提示，引导，让你先改，不要自己一股脑的全部操作完，你是研究者，不是打工人
7. **关键步骤可解释**：对于非bug修改的操作，务必新建运行文件夹进行日志记录，对实验结果进行严谨分类和记录

## 质量检查
- 每个模块完成后我会主动做 review 检查逻辑错误
- 报告是你的核心产出物，我会帮拟框架、改表述，但内容需要你来定
- 截止前做最终完整性检查

## 记忆与上下文
- 我会把项目决策、关键问题、你学到的知识点记录在 memory/ 目录
- 如果我对某个做法不理解，我会先问清楚再做，不猜
- 任何我想改 plan.md 的操作，都会先跟你确认

## 快速命令
```bash
uv init                        # 初始化项目
uv add <package>               # 添加依赖
uv sync                        # 同步环境
uv run python src/train.py     # 在虚拟环境中运行
```
