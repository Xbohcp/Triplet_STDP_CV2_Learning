# 算法实现说明

本文档说明 `Coding_prompt.tex` 中的公式如何映射到代码。

## 1. 网络与前向传播

代码中的网络是 `PositiveMLP`：

```text
784 -> 500 -> 500 -> 500 -> 10
```

每一层都执行：

```text
x_l = W_l y_(l-1) + b_l
y_l = sigma(x_l)
```

其中：

```text
sigma(x) = softplus(x) + 1e-6
sigma'(x) = sigmoid(x)
```

这样所有 `y_l` 都严格为正，满足需求中“激活函数的值局限在正值范围”的条件。

## 2. 分类损失

因为最后一层输出也是正值激活，不直接把它当作未约束 logits。代码使用：

```text
logits = log(y_L)
loss = CrossEntropyLoss(logits, target)
```

这样既保持输出正值约束，又可以使用标准分类交叉熵。

## 3. 标准 BP 权重变化

标准 BP 更新由 PyTorch autograd 计算：

```text
Delta W_l^BP = -eta * dL/dW_l
```

对应函数：

```python
bp_weight_deltas(model, loss, learning_rate)
```

## 4. 生物合理误差传播

输出层误差先取：

```text
epsilon_L = dL/dx_L
```

隐藏层误差使用需求中的递推式：

```text
epsilon_l =
  zeta(
    y_l^-1
    * sigma'(x_l)
    * W_(l+1)^T (y_(l+1) * epsilon_(l+1))
  )
```

在 batch 形式下，代码用矩阵乘法：

```python
propagated = (y_next * epsilon_next) @ W_next
raw_error = propagated * sigma_prime(x_l) / y_l
epsilon_l = zeta(raw_error)
```

其中默认：

```text
zeta(x) = 0.5 * tanh(x)
```

也可通过 `--scale-mode clamp` 使用硬截断到 `(-0.5, 0.5)` 附近。

对应函数：

```python
biological_errors(model, cache, targets, scale_mode)
```

## 5. 生物合理权重更新

单样本公式为：

```text
Delta W_l = -eta * f(epsilon_l, y_l) * y_(l-1)^T
```

batch 形式使用平均更新：

```text
Delta W_l = -(eta / batch_size) * F_l^T @ y_(l-1)
F_l = f(epsilon_l, y_l)
```

默认：

```text
f(epsilon_l, y_l) = epsilon_l * y_l
```

可选：

- `epsilon_times_activation`：`epsilon_l * y_l`
- `epsilon`：只使用 `epsilon_l`
- `activity_gated`：用平均活动强度归一化后的活动门控

对应函数：

```python
biological_weight_deltas(...)
```

## 6. 夹角统计

对每一层，将两个学习规则产生的权重变化矩阵展平成向量：

```text
theta_l = arccos(
  <Delta W_l^BP, Delta W_l^bio>
  / (||Delta W_l^BP|| * ||Delta W_l^bio||)
)
```

然后对所有有效层取平均，得到当前 batch 的平均夹角。一个 epoch 内再对 batch 取平均，输出：

- `mean_angle_degrees`
- `layer_1_angle_degrees`
- `layer_2_angle_degrees`
- `layer_3_angle_degrees`
- `layer_4_angle_degrees`

如果某层出现零向量，夹角记为 `None`，不会参与平均。

## 7. 实验解释建议

建议先运行两组实验：

```bash
python -m triplet_stdp_cv2_learning.train_mnist --train-rule biological --epochs 20
python -m triplet_stdp_cv2_learning.train_mnist --train-rule bp --epochs 20 --output-csv outputs/bp_train_metrics.csv --output-plot outputs/bp_train_angle.png
```

两者的区别是网络轨迹不同：

- `--train-rule biological`：观察生物规则训练出的参数轨迹上，生物更新方向与 BP 方向的夹角。
- `--train-rule bp`：观察标准 BP 训练出的参数轨迹上，生物更新方向与 BP 方向的夹角。

如果目标是检验生物规则能否逼近 BP，重点看平均夹角是否随 epoch 下降，以及不同层的夹角是否同步下降。
