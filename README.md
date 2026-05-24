# Triplet STDP CV2 Learning

本项目根据 `Coding_prompt.tex` 中的需求，提供一个 PyTorch 实验框架，用 MNIST 对比两种学习算法产生的权重变化向量夹角：

1. 标准反向传播：`Delta W_l = -eta * dL/dW_l`
2. 正值激活下的生物合理误差传播与权重更新：

```text
epsilon_l = zeta(y_l^-1 * sigma'(x_l) * W_(l+1)^T (y_(l+1) * epsilon_(l+1)))
Delta W_l = -eta * f(epsilon_l, y_l) * y_(l-1)^T
```

默认网络为 3 个隐藏层，每层 500 个神经元，数据集为 MNIST。

## 安装

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速运行

推荐直接使用一键脚本：

```bash
./scripts/run_experiment.sh --smoke
./scripts/run_experiment.sh --epochs 20 --open
```

脚本会自动创建 `.venv`、安装依赖、运行测试，然后执行实验。查看所有选项：

```bash
./scripts/run_experiment.sh --help
```

完整实验：

```bash
python -m triplet_stdp_cv2_learning.train_mnist \
  --epochs 20 \
  --batch-size 128 \
  --learning-rate 0.01 \
  --train-rule biological \
  --output-csv outputs/angle_metrics.csv \
  --output-plot outputs/angle_trend.png
```

快速 smoke run：

```bash
python -m triplet_stdp_cv2_learning.train_mnist \
  --epochs 2 \
  --limit-samples 1024 \
  --output-csv outputs/smoke_metrics.csv \
  --output-plot outputs/smoke_angle_trend.png
```

运行后会生成：

- `outputs/angle_metrics.csv`：每个 epoch 的 loss、accuracy、平均夹角、各层夹角。
- `outputs/angle_trend.png`：平均夹角随 epoch 变化的趋势图，同时包含各层曲线。

## 关键实验设定

- 平台：PyTorch
- 数据集：MNIST，首次运行自动下载到 `data/`
- 网络结构：`784 -> 500 -> 500 -> 500 -> 10`
- 激活函数：所有线性层后使用 `softplus(x) + eps`，保证激活值为正
- 分类损失：对正值输出取 `log(y)` 作为 logits 后使用 cross entropy
- 默认截断函数：`zeta(x) = 0.5 * tanh(x)`
- 默认更新函数：`f(epsilon_l, y_l) = epsilon_l * y_l`
- 夹角单位：degree

## 可调参数

```bash
python -m triplet_stdp_cv2_learning.train_mnist --help
```

常用参数：

- `--train-rule biological|bp|none`：选择实际用于更新网络参数的规则。无论选择哪种，脚本都会在同一批次、同一网络状态上计算 BP 与生物规则的权重变化向量并比较夹角。
- `--scale-mode tanh|clamp`：选择 `zeta` 的实现。
- `--update-function epsilon_times_activation|epsilon|activity_gated`：选择 `f(epsilon_l, y_l)`。
- `--update-bias`：是否同步更新 bias。权重夹角比较始终只使用 weight。
- `--limit-samples N`：只使用 MNIST 前 N 个样本，便于调试。

## 测试

```bash
python -m pytest tests/test_learning_rules.py -q
```

测试覆盖：

- 零向量夹角返回 `None`，避免除零。
- 正交向量夹角为 90 度。
- BP 与生物规则生成的每层权重变化张量形状一致。

## 文件结构

- `triplet_stdp_cv2_learning/model.py`：正值激活 MLP 和 forward cache。
- `triplet_stdp_cv2_learning/learning_rules.py`：BP 更新、生物误差传播、生物更新、夹角统计。
- `triplet_stdp_cv2_learning/train_mnist.py`：MNIST 实验入口，保存 CSV 和趋势图。
- `tests/test_learning_rules.py`：核心规则的单元测试。
- `docs/algorithm_details.md`：公式到代码的详细对应关系。
