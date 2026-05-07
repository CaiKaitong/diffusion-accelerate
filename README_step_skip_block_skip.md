# 当前 Step-Skip 和 Block-Skip 计算逻辑说明

这份文档说明的是**当前代码里实际实现的逻辑**。

目前仓库中的 `step-skip` 已经统一改成了 **AdaptiveDiffusion 风格** 的跳步逻辑，旧的基于 action 变化比例的打分逻辑已经移除。

主要对应以下代码：

- `diffusion_policy/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py`
- `diffusion_policy/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py`
- `BACInfer/core/diffusion_cache_wrapper.py`

## 1. 总体区别

当前代码里有两层“跳过”机制：

1. `step-skip`
2. `block-skip`

它们不是一回事。

### 1.1 step-skip

- 发生在 diffusion 采样循环外层
- 决定“这一整个 denoising step 要不要重新跑一次 model forward”
- 如果触发跳过，就直接复用上一个 step 的 `model_output`

### 1.2 block-skip

- 发生在 transformer 模型内部
- 决定某个 `TransformerDecoderLayer` 里的子模块要不要重算
- 如果触发跳过，就复用这个 block 上一次缓存下来的输出

所以：

- `step-skip` 跳的是**整个 diffusion step**
- `block-skip` 跳的是**step 内部某个 transformer 子块**

---

## 2. step-skip 现在怎么计算

现在的 `step-skip` 逻辑来自 `AdaptiveDiffusion` 的思路，不再使用旧版本那种“根据 action 变化量拼一个 score”的做法。

当前逻辑本质上是：

- 跟踪连续几个 timestep 的 latent 变化幅度
- 用一个“三阶差分近似误差”判断当前轨迹是否足够平滑
- 如果足够平滑，就允许下一步直接复用上一次的 `model_output`

控制参数是：

- `enable_skip`
- `skip_score_threshold`
- `skip_max_stale`
- `skip_min_progress`

其中：

- `skip_score_threshold` 现在更准确地说是 **AdaptiveDiffusion 风格的误差阈值**
- `skip_max_stale` 现在更准确地说是 **允许连续跳过的最大步数**

当前仓库默认固定为：

- `skip_score_threshold = 0.03`
- `skip_max_stale = 4`

---

## 3. step-skip 依赖哪些状态

每次采样时，代码维护以下状态：

- `prev_model_output`
- `prev_latents`
- `diff_list`
- `compute_mask`

含义分别是：

- `prev_model_output`：上一次真正执行 model forward 得到的输出
- `prev_latents`：历史若干步的 latent
- `diff_list`：相邻 latent 之间的平均绝对差
- `compute_mask`：记录“下一步应该重算还是允许跳过”的布尔决策

---

## 4. latent 差分怎么定义

对于相邻两步 latent，代码用下面的量衡量变化幅度：

```python
(latent_t - latent_{t-1}).abs().mean()
```

也就是：

- 先做逐元素绝对值差
- 再对所有元素取均值

这和旧逻辑完全不同。  
旧逻辑看的是 action 之间的 L2 norm，现在看的是 latent 之间的平均绝对差。

---

## 5. 前几步为什么不会直接跳

当前实现会先积累足够的历史 latent。

只有当：

- 已经有足够多历史 latent
- 并且已经能构造出最近三段差分关系

之后，才会开始根据估计器决定下一步能不能 skip。

这也是为什么刚开始几步通常还是会正常算 forward。

---

## 6. AdaptiveDiffusion 风格的核心判断

假设当前我们已经有三段相关差分：

- `prev_prev_diff`
- `prev_diff`
- `cur_diff`

其中：

- `prev_prev_diff`：更早一段的 latent 差分
- `prev_diff`：上一段 latent 差分
- `cur_diff`：当前 latent 相对前一个 latent 的差分

代码用下面这个量来判断局部变化是否平滑：

```python
abs((cur_diff + prev_prev_diff) / 2 - prev_diff)
```

这可以理解成：

- 用前后两段差分的平均值去近似中间那段差分
- 看中间那段和这个近似值偏差有多大

如果偏差足够小，就说明最近几步 latent 变化比较平滑，下一步可以更放心地跳过。

---

## 7. 什么时候允许下一步跳过

当前代码的判断是：

```python
abs((cur_diff + prev_prev_diff) / 2 - prev_diff) <= prev_diff * skip_score_threshold
```

如果这个条件成立，就会把“下一步允许跳过”的标记写成：

```python
should_compute_next = False
```

否则：

```python
should_compute_next = True
```

换句话说：

- **误差小于阈值** -> 下一步可以 skip
- **误差大于阈值** -> 下一步要重新计算

这里的阈值是相对阈值，不是绝对阈值，因为右边是：

```python
prev_diff * skip_score_threshold
```

所以 `skip_score_threshold` 越大，skip 条件越宽松；越小，skip 条件越严格。

---

## 8. `skip_max_stale` 现在是什么意思

当前实现里，`skip_max_stale` 不再是旧逻辑里那种简单的 `stale` 计数器比较。

现在它的作用是：

- 限制最近连续若干步不能全部都是 skip

对应判断是：

```python
if len(compute_mask) > 4 and not any(compute_mask[-skip_max_stale:]):
    should_compute_next = True
```

这里的含义是：

- `compute_mask` 里 `True` 表示该步应当重算
- `False` 表示该步允许跳过
- 如果最近 `skip_max_stale` 步里一个 `True` 都没有
- 那么下一步强制设为 `True`

也就是：

- 不允许无限连续跳
- 跳过若干步之后，必须插入一次真实计算

这和 AdaptiveDiffusion 里 “max skip steps” 的意思是一致的。

---

## 9. step-skip 在实际执行时怎么用

在每个 timestep 开始时，代码会先看上一轮记录好的 `compute_mask[-1]`：

```python
should_compute = compute_mask[-1]
use_cache = not should_compute
```

如果 `use_cache == True`：

```python
model_output = prev_model_output
```

如果 `use_cache == False`：

```python
model_output = model(trajectory, t, cond)
prev_model_output = model_output.detach()
```

所以整个机制是：

1. 当前步先执行“上一时刻已经决定好的 skip/compute 决策”
2. 当前步完成后，再根据新的 latent 差分去决定“下一步”要不要 skip

---

## 10. `skip_min_progress` 还在做什么

当前实现里仍然保留了：

```python
progress = float(t.item()) / max(1, self.num_inference_steps - 1)
```

以及：

```python
progress >= skip_min_progress
```

它的作用还是一样：

- 只有采样进度达到某个比例以后，才允许 step-skip 生效

如果 `skip_min_progress = 0.0`，那就表示从允许阶段一开始就可以启用 AdaptiveDiffusion 风格判断。

---

## 11. step-skip 的核心直观理解

现在的 step-skip 不再问：

- “最近几次 action 变化比例像不像可以跳？”

而是改成问：

- “最近几步 latent 的变化轨迹是否足够平滑，以至于下一步可以直接复用上一次噪声预测？”

所以现在更准确的理解方式是：

- 这是一个 **基于 latent 差分平滑性的 adaptive skip 逻辑**

而不是：

- 一个基于 action 变化启发式打分的 skip 逻辑

---

## 12. block-skip 现在怎么算

严格来说，**当前代码里的 block-skip 并没有在线计算一个类似 step-skip 的显式相似度分数**。

block-skip 的核心依据仍然是：

- `cache_age`
- 每个 block 的阈值
- 可选的自适应阈值更新

### 12.1 block 是怎么切开的

在 `BACInfer/core/diffusion_cache_wrapper.py` 里，代码会把每个 `TransformerDecoderLayer` 拆成 3 个可缓存子块：

- `*_sa_block`：self-attention
- `*_mha_block`：cross-attention
- `*_ff_block`：feed-forward

每个 block 单独决定：

- 重算
- 还是复用缓存

### 12.2 block 维护了哪些状态

代码为每个 block 维护：

- `last_update_steps[block_key]`
- `ema_ages[block_key]`
- `current_thresholds[block_key]`

还维护一个全局的：

- `current_step`

### 12.3 cache_age 是什么

block-skip 最核心的量是：

```python
cache_age = current_step - last_update_step
```

意思是：这个 block 距离上一次真正重算，已经过了多少 step。

如果一个 block 还从来没算过缓存，那第一次一定会强制更新。

---

## 13. 非自适应 block-skip 规则

如果 gate config 不是自适应模式，那么每个 block 会从配置里读取：

- `threshold`
- `direction`
- `max_age`

对应逻辑大致是：

```python
if max_age is not None and cache_age >= max_age:
    update = True
elif direction == "high->update":
    update = cache_age >= threshold
elif direction == "low->update":
    update = cache_age <= threshold
else:
    update = True
```

也就是说，当前 block-skip 的本质是：

**看缓存年龄有没有到阈值**

而不是：

**在线算当前激活和历史激活的相似度，再决定要不要更新**

---

## 14. 自适应 block-skip 规则

如果配置里有：

```python
gate["mode"] == "adaptive_cache_age"
```

那么 block-skip 仍然不是在线算激活相似度分数，而是：

1. 先看 `cache_age`
2. 再用历史 `cache_age` 的 EMA 来动态调整阈值

### 14.1 初始阈值

第一次会先初始化当前 block 的阈值：

```python
init_threshold
```

如果没有，就退化成：

```python
min_threshold
```

### 14.2 EMA 怎么更新

每当 block 真正发生一次更新，代码会用这次更新前的 `cache_age` 去更新 EMA：

```python
new_ema = age_value if prev_ema is None else (1.0 - eta) * prev_ema + eta * age_value
```

其中：

- `eta = ema_eta`

### 14.3 新阈值怎么得到

然后用 EMA 算一个新的阈值：

```python
new_threshold = round(gamma * new_ema)
new_threshold = max(min_threshold, min(max_threshold, float(new_threshold)))
```

其中：

- `gamma = threshold_gamma`
- `min_threshold` / `max_threshold` 用来做上下界裁剪

### 14.4 自适应模式下的更新判断

当前判断大致是：

```python
if max_threshold is not None and cache_age >= max_threshold:
    update = True
elif cache_age is None:
    update = True
else:
    update = cache_age >= current_threshold
```

所以 block-skip 本质上仍然是：

- **age-based gating**

而不是：

- **similarity scoring**

---

## 15. step-skip 和 block-skip 的本质差异

### 15.1 step-skip

当前是：

- 基于 latent 差分平滑性的 AdaptiveDiffusion 风格逻辑
- 决定整次 model forward 是否复用
- 用 `skip_score_threshold` 控制误差容忍度
- 用 `skip_max_stale` 控制最大连续跳步数

### 15.2 block-skip

当前是：

- 基于 `cache_age`
- 决定 block 级别是否重算
- 可以带自适应阈值
- 没有对应的在线 latent/action 相似度分数公式

---

## 16. 当前仓库的结论

如果你现在以代码实现为准，那么：

- `step-skip` 已经统一为 **AdaptiveDiffusion 风格**
- 旧的 action-based step score 逻辑已经移除
- `block-skip` 仍然是基于 cache age 的 block cache reuse / update gating

因此当前最准确的理解是：

- 外层：AdaptiveDiffusion-style `step-skip`
- 内层：cache-age-based `block-skip`
