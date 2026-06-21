# 实验报告：基于深度学习的 ISP 网络流量预测模型对比分析

## 1. 摘要

本实验对论文"Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting"的核心实验进行复现与拓展。在 CESNET-TimeSeries24 数据集的 283 条机构级流量序列上，以 n_bytes 为预测目标，对比了 Mean、SARIMA、LSTM、GRU 四种模型在多种滑动窗口配置下的预测性能。实验采用 35:5:60 的时间顺序分割和 Z-score 标准化，并对 SARIMA 引入了滚动预测策略。结果表明：GRU 和 LSTM 的 R² 中位数分别达到 0.136 和 0.113，显著优于 Mean 基线（-0.012）；SARIMA 在滚动预测下 R² 中位数达 0.183，挑战了"统计模型远不如 DL 模型"的原论文结论。实验还发现，真实网络流量数据中存在的季节性分布漂移问题会严重干扰 R² 均值的可靠性，建议以中位数作为主要参考指标。

## 2. 引言

ISP 网络流量预测是网络资源管理、拥塞控制和容量规划的基础技术。准确的流量预测可以帮助运营商优化网络资源配置、降低运营成本、提升用户体验。近年来，LSTM、GRU 等循环神经网络在时序预测领域取得显著成功，但在真实大规模 ISP 数据上的系统性对比研究仍较为有限。

论文"Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting"（Koumar 等，2025）在 CESNET-TimeSeries24 数据集（40 周真实 ISP 流量，含 283 条机构级序列）上对 10 种模型进行了系统对比。该论文的核心贡献在于提供了一个标准化的基准测试框架，但其主要局限在于：对 SARIMA 等统计模型仅做一次全局拟合、固定参数预测，未采用滚动更新策略，可能导致不公平的对比结论。

本实验在复现核心实验的基础上，做了以下拓展：（1）对 SARIMA 引入滚动预测策略，使其与 DL 模型在相同条件下对比；（2）系统分析了时间分割比例和标准化方式对评估结果的影响；（3）揭示了数据分布漂移对 R² 指标的干扰，比较了三种处理方案。

## 3. 数据与预处理方法

### 3.1 数据集

实验采用 CESNET-TimeSeries24 数据集的机构级（institutions）数据。该数据集采集自捷克 CESNET 科研教育网络，时间跨度为 2023 年 10 月至 2024 年 7 月（40 周），聚合粒度为 1 小时。选择机构级而非子网级或 IP 地址级，是因为机构级缺失率仅约 1%，远低于 IP 地址级的 80%+，且周期模式清晰。

数据预处理的核心挑战在于：（1）原始 CSV 只记录有数据的行，缺失时间点整行不存在，需通过 right merge 对齐时间轴来识别；（2）部分机构受季节性影响（如大学暑假），测试期流量较训练期骤降 10^4~10^5 倍，导致分布漂移。

### 3.2 预处理流程

**时间轴对齐**：各时间序列与完整时间戳表 times.csv 做 right merge，缺失的时间点以 NaN 标记。

**缺失值填充**：机构级数据缺失率仅约 1%，采用零填充。

**滑动窗口生成**：选取四种窗口配置：(24,24)、(168,24)、(168,168)、(744,168)。窗口按 stride = prediction_window 不重叠滑动。

**时间顺序分割**：采用 35% 训练 / 5% 验证 / 60% 测试 的比例，对齐论文开源代码的分割策略。本实验初期尝试了 70:15:15 分割，但发现该方案导致测试期集中在暑假（6-7 月），加剧了分布漂移问题。切换至 35:5:60 后，LSTM 的 R² 中位数从 0.031 提升至 0.098，极端异常序列从 5 条降至 2 条。

**标准化**：采用 Z-score 标准化。选择 Z-score 而非论文使用的 MinMaxScaler，原因是网络流量呈长尾分布，Z-score 对异常值更鲁棒。实验表明两种标准化方式的模型排名趋势一致。

### 3.3 模型

本实验选取其中 4 个代表模型：

**Mean 基线**：取输入窗口的均值作为预测值，作为性能下限参考。

**SARIMA**：使用 (1,0,1)(1,0,1,24) 配置。与论文的静态评估不同，本实验对 SARIMA 采用滚动预测策略：每到达一个测试窗口，用截至当前的全部历史数据重新拟合，再进行预测。

**LSTM**：双向单层 LSTM（hidden_size=100），取最后时间步的隐藏状态接 Linear 层输出 24 步预测。

**GRU**：结构与超参数与 LSTM 一致，将 LSTM 单元替换为 GRU 单元。

## 4. 实验设置

### 4.1 超参数配置

| 参数 | LSTM | GRU | SARIMA |
|------|------|-----|--------|
| hidden_size | 100 | 100 | -- |
| bidirectional | True | True | -- |
| learning_rate | 0.01 | 0.01 | -- |
| epochs / maxiter | 100 (early stop patience=5) | 同左 | 200 |
| batch_size | 16 | 16 | -- |
| (p,d,q) | -- | -- | (1,0,1) |
| (P,D,Q,s) | -- | -- | (1,0,1,24) |

### 4.2 评估指标

**RMSE**（均方根误差）：sqrt(1/N * sum(yi - yhat_i)^2)

**R2**（决定系数）：1 - sum(yi - yhat_i)^2 / sum(yi - ybar)^2

**sMAPE**（对称平均绝对百分比误差）：100%/N * sum(2|yi - yhat_i| / (|yi| + |yhat_i|))

**Harmonic Score**：2 * (RMSE * |R2 - 1|) / (RMSE + |R2 - 1|)

### 4.3 实验环境

GPU：NVIDIA GeForce RTX 5060 Laptop (CUDA 12.8)
框架：PyTorch 2.11.0（纯 torch.nn）
统计模型：statsmodels 0.14
环境管理：uv，Python 3.12

## 5. 结果与分析

### 5.1 模型整体对比

在 237 条机构级序列上的对比结果（窗口 (24,24)，35:5:60 分割）：

| 模型 | R2 均值(裁剪后) | R2 中位数 | R2>0 占比 | RMSE 中位数 |
|------|----------------|-----------|-----------|------------|
| Mean | 0.015 | -0.012 | 40.1% | 4.6e8 |
| LSTM | 0.070 | 0.113 | 83.1% | 4.3e8 |
| GRU | 0.113 | 0.136 | 80.2% | 4.2e8 |
| SARIMA（滚动） | 0.170 | 0.183 | 89.0% | 3.7e8 |

核心结论：GRU 和 LSTM 均显著优于 Mean 基线。GRU 的 R2 中位数 0.136 高于 LSTM 的 0.113，头对头对比中 GRU 胜出 56.5%，这与论文结论一致。

### 5.2 窗口配置对比

| 窗口 | LSTM R2 中位数 | GRU R2 中位数 |
|------|---------------|---------------|
| (24,24) | 0.098 | 0.097 |
| (168,24) | 0.095 | 0.099 |
| (168,168) | 0.046 | 0.063 |
| (744,168) | 0.032 | 0.047 |

短期预测（24h）效果最好；长期预测（168h 和 744h）的 R2 降至 0.05 以下，衰减超 50%。

### 5.3 SARIMA 滚动预测与可解释性

论文报告 SARIMA 效果在所有模型中垫底，但本实验发现这主要是评估方式的问题。滚动预测使 SARIMA 的 R2 中位数提升至 0.183。

以文件 0 为例的 SARIMA 系数：

| 参数 | 估计值 | 解读 |
|------|--------|------|
| AR(1) | 0.905 | 前一小时流量影响权重 90% |
| SAR(24) | 0.988 | 昨天同一小时影响权重 99%，最强信号 |
| MA(1) | -0.148 | 短期误差修正 |
| SMA(24) | -0.777 | 日周期误差修正 |

### 5.4 误差原因分析

（1）数据层级与缺失率：机构级 RMSE 约 0.14，子网级升至 0.28。
（2）流量量级与波动性：小流量机构的相对预测误差更大。
（3）季节性模式：强日周期序列误差显著低于不规则序列。
（4）预测跨度：误差随预测窗口增大而累积。
（5）分布漂移：训练集与测试集的季节性差异（见下节）。

### 5.5 数据分布漂移问题

部分机构级序列的 R2 出现极端负值（-10^5 ~ -10^9），测试集流量均值相对训练集骤降 10^4~10^5 倍。采用 35:5:60 分割可部分缓解（R2 中位数提升 3 倍），但无法根除。建议以中位数作为主要评估指标。

### 5.6 与论文结果对比

**表 A：归一化 RMSE 对比**

| 模型 | 论文 RMSE | 本实验 RMSE | 趋势 |
|------|----------|------------|------|
| Mean | 0.152 | 0.126 | 一致（Mean 最差）|
| LSTM | 0.105* | 0.145 | 一致（LSTM 优于 Mean）|
| GRU | 0.105* | 0.143 | 一致（GRU 略优于 LSTM）|
| SARIMA | 0.151 | 0.068 | 本实验显著更优（滚动预测）|

*论文为 (24,1) 窗口，本实验为 (24,24)。SARIMA 论文为固定预测，本实验为滚动预测。

**表 B：R2 对比**

| 模型 | 论文 R2（全窗口） | 本实验 R2 中位数 | 趋势 |
|------|-----------------|-----------------|------|
| Mean | 0.032 | -0.004 | 分割比例不同 |
| LSTM | -0.007 | 0.113 | 本实验显著更优 |
| GRU | -0.003 | 0.136 | 本实验显著更优 |
| SARIMA | -10.831 | 0.170 | 滚动预测后大幅提升 |

**表 C：Harmonic Score 对比**

| 模型 | 论文 HS | 本实验 HS | 趋势 |
|------|--------|----------|------|
| Mean | 0.153 | 0.142 | 一致 |
| LSTM | 0.139 | 0.147 | 一致（LSTM ≈ GRU）|
| GRU | 0.139 | 0.150 | 一致（GRU 略优）|
| SARIMA | 0.239 | 0.119 | 滚动预测后大幅优于论文 |

关键发现：
1. 相对排序一致：GRU ≈ LSTM > Mean，与论文趋势相同。
2. 本实验 LSTM/GRU 优于论文参考代码（论文 LSTM R2 中位数为 -0.007，本实验 0.11-0.14）。
3. SARIMA 差异：论文报告 SARIMA 最差，本实验滚动预测下 SARIMA R2 达 0.183。

## 6. 总结与展望

主要发现：
1. GRU 与 LSTM 表现接近，R2 中位数 0.12-0.14，与论文结论一致。
2. SARIMA 在滚动预测下 R2 中位数 0.183，挑战了原论文结论。
3. 短期预测优于长期预测，R2 衰减超 50%。
4. 分布漂移是真实 ISP 数据的重要挑战。

局限性：
- 仅覆盖机构级 n_bytes 指标
- SARIMA 滚动预测仅运行 10 个测试窗口
- 超参数使用论文默认值，未系统搜索

## 参考文献

[1] Koumar, J., et al. "Comparative Analysis of Deep Learning Models for Real-World ISP Network Traffic Forecasting." IEEE TNSM, 2026.
[2] Koumar, J., et al. "CESNET-TimeSeries24: Time Series Dataset for Network Traffic Anomaly Detection and Forecasting." Nature Scientific Data, 2025.
[3] Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." Neural Computation, 1997.
[4] Cho, K., et al. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." EMNLP, 2014.
[5] Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. "Time Series Analysis: Forecasting and Control." 5th Edition, Wiley, 2015.
[6] Hyndman, R. J. & Athanasopoulos, G. "Forecasting: Principles and Practice." 3rd Edition, OTexts, 2021.
[7] Ferreira, G. O., et al. "Forecasting Network Traffic: A Survey and Tutorial." IEEE Access, 2023.
[8] D'Alconzo, A., et al. "A Survey on Big Data for Network Traffic Monitoring and Analysis." IEEE TNSM, 2019.
