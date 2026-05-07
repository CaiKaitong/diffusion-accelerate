# Block-Skip Discussion

这个文件单独用来记录当前仓库里关于 `block-skip` 的讨论、设计选择、实验顺序和结论。

## 1. 当前共识

### 1.1 step-skip 完全固定

后续讨论 `block-skip` 时，默认 `step-skip` 不再改动。

当前固定版本：

- 判定逻辑：`AdaptiveDiffusion` 风格的 latent-based skip score
- skip 行为：真正 skip 时，直接复用上一轮 `noise_pred / model_output`
- 不做 delta compensation

当前固定参数：

- `skip_score_threshold = 0.03`
- `skip_max_stale = 4`

这部分已经视为基线，后续所有 block-skip 改动都应建立在这套固定 step-skip 之上。

## 2. 为什么要改 block-skip

当前 block-skip 主要还是基于 `cache_age` 决定是否更新缓存。

这类规则有一个明显问题：

- 它只知道“距离上次更新过去了几步”
- 它不知道“当前特征到底变了多少”

所以会出现两类误差：

1. 特征几乎没变，但因为 age 到阈值了，被迫更新
2. 特征已经明显变了，但因为 age 还没到阈值，仍然错误复用

因此后续的目标是：

- 把 block-skip 的打分改成和特征变化直接相关
- 让缓存复用更准确，而不是只靠时间步年龄控制

## 3. 设计约束

为了保证实验变量干净，目前先遵守下面几个约束：

1. 不动 step-skip
2. 先只改 block-skip
3. 新逻辑优先做成增量式，不直接删掉旧的 age-based 方案
4. 第一版先追求可解释、稳定、容易可视化，不追求一上来最复杂

## 4. 第一版 block-skip 的建议方向

### 4.1 核心思路

不要只看 `cache_age`，而是看“当前 block 输入特征”和“上次缓存对应的 block 输入特征”之间的变化程度。

也就是说，block 是否应该更新，应该由 feature drift 决定。

### 4.2 为什么先看 block 输入，而不是 block 输出

优先看 block 输入更合理，原因有两个：

1. 这是 block 当前要处理的真实上下文
2. 如果输入几乎没变，复用上一次 block 输出才更有依据

如果直接拿 block 输出去比，会把“当前到底该不该运行这个 block”这个判断，建立在旧输出上，解释上会更绕。

## 5. 第一版建议先从 ff_block 开始

建议先只在 `ff_block` 上做 feature-aware block-skip，原因：

1. `ff_block` 接口最简单
2. 没有 cross-attention 那种额外依赖
3. 更适合先验证“feature drift score 是否有效”
4. 出问题时更容易定位

等 `ff_block` 跑通并且可视化结果合理，再考虑扩展到：

- `sa_block`
- `mha_block`

## 6. 第一版分数定义建议

第一版先用最简单的相对 L2 变化：

```text
score = ||x_cur - x_ref|| / (||x_ref|| + eps)
```

其中：

- `x_cur`：当前步进入该 block 前的输入特征
- `x_ref`：上一次缓存更新时记录的输入特征
- `eps`：防止分母为 0

判定方式：

- `score < threshold`：复用缓存
- `score >= threshold`：更新 block

同时保留一个 `max_age` 兜底：

- 如果一个 block 太久没有更新，即便 score 小，也强制更新一次

## 7. 为什么第一版不建议一开始就上复杂混合分数

例如：

- cosine similarity
- L1 + L2 混合
- attention-aware score
- 多统计量组合

这些后面都可以试，但不适合第一步就上。原因：

1. 变量太多，不利于判断收益来自哪里
2. 阈值更难调
3. 可视化后不容易解释
4. 如果结果不好，不容易判断是实现问题还是打分设计问题

所以第一版建议先从：

- 单 block 类型
- 单一分数
- 单一阈值

开始。

## 8. 建议保留的兜底机制

即使进入 feature-aware block-skip，也建议保留一些硬约束：

### 8.1 max_age

避免一个 block 长时间不更新。

### 8.2 首次必更新

如果没有缓存，必须先算一次。

### 8.3 可能的 warmup 区间

前几步 diffusion 变化通常更大，必要时可以考虑前若干步不启用 block-skip。

这个是否需要，后面看可视化结果再决定。

## 9. 当前推荐的实验顺序

建议按下面顺序推进：

1. 固定 step-skip，不再改
2. 给 block-skip 增加 feature trace
3. 在 `ff_block` 上实现 feature-aware score
4. 保留 `max_age` 作为兜底
5. 用单样本推理脚本输出 block reuse 矩阵
6. 先看 `can_mh`
7. 再看 `square_mh`
8. 如果矩阵表现合理，再做 success rate / speedup 对比

## 10. 当前需要记录和可视化的内容

后续 block-skip 改造时，建议至少记录：

- diffusion step index
- block key
- 本步是否复用
- 本步是否更新
- 当前 `cache_age`
- 当前 feature score
- 当前阈值
- 是否因为 `max_age` 被强制更新

这样后面看矩阵图和 json trace 时，才知道每一次 skip 到底为什么发生。

## 11. 暂时不做的事情

当前先不做：

- 修改 step-skip 逻辑
- 引入 delta compensation
- 一上来同时改所有 block
- 一上来做复杂混合打分
- 一上来把 age-based 逻辑完全删掉

## 12. 下一步执行建议

最近一步建议是：

1. 单独保留这个讨论文档
2. 开始在 `diffusion_cache_wrapper.py` 里加入 feature-aware block score 分支
3. 第一版只接 `ff_block`
4. 然后用已经写好的 `visualize_skip_matrix.py` 跑单样本，看 block reuse 矩阵是否合理

## 13. 待补充

后续可以继续往这个文件里补：

- 具体公式版本
- 阈值扫描结果
- 不同 task 的可视化观察
- 最终采用的 block-skip 实现
- 被放弃的方案和原因

## 14. 2026-05-05 第一次 feature-aware block-skip 可视化观察

这次已经做了一版真正可运行的 feature-aware block-skip：

- 三类 block 全部一起接入：
  - `sa_block`
  - `mha_block`
  - `ff_block`
- score 使用相对 L2 drift
- 仍然保留 `max_age`
- step-skip 保持固定不动

### 14.1 本次实验配置

任务：

- `can_mh`

模式：

- `hybrid`

step-skip 固定参数：

- `skip_score_threshold = 0.03`
- `skip_max_stale = 4`

block-skip gate：

- `mode = feature_distance`
- 三类 block 都使用：
  - `threshold = 0.035`
  - `max_age = 6`
  - `reduce = mean`
  - `eps = 1e-6`

### 14.2 结果现象

单样本矩阵结果显示：

- `num_blocks = 24`
- `num_block_trace_steps = 100`
- `block_reuse_ratio ≈ 0.00137`

也就是：

- 几乎所有 block 在几乎所有 diffusion step 都选择了更新
- 真正发生的 block reuse 极少

统计上大致是：

- `reuse = 2`
- `update = 1462`

### 14.3 score 分布观察

这次记录到的 score 分布大致为：

- `min ≈ 0.031`
- `p25 ≈ 0.134`
- `p50 ≈ 0.217`
- `p75 ≈ 0.333`
- `p90 ≈ 0.475`
- `max ≈ 1.155`

这说明：

- 当前 `threshold = 0.035` 太低了
- 它几乎贴近 score 分布的最小值
- 所以绝大多数 block 都会被判定为“需要更新”

### 14.4 当前结论

这版逻辑本身是通的：

- trace 正常
- score 正常记录
- 三类 block 都真正接入了 feature-aware 判定

但是这版参数下，矩阵形态还不像一个有效的 skip 逻辑，因为：

- 几乎没有形成成片的 reuse 区域
- 整体更像“几乎全量重算”

所以接下来的重点不是继续怀疑接口是否通，而是：

- 调整 feature threshold 的量级
- 让 block reuse 矩阵先呈现出更合理的结构

### 14.5 下一步建议

下一步优先做：

1. 先不改 step-skip
2. 保持 feature-aware block-skip 框架不变
3. 直接针对 block threshold 做一小圈人工试探
4. 先看矩阵形状，不先看最终任务分数

因为当前 score 的中位数在 `0.21` 左右，所以后续更合理的第一轮尝试区间，应该明显高于 `0.035`。

## 15. 2026-05-05 block threshold 第一轮扫描

在保持 step-skip 固定不动的前提下，已经对 `can_mh` 做了一轮单样本 block threshold 扫描。

### 15.1 固定条件

- task: `can_mh`
- mode: `hybrid`
- step-skip:
  - `skip_score_threshold = 0.03`
  - `skip_max_stale = 4`
- block-skip:
  - `mode = feature_distance`
  - 三类 block 使用相同 threshold
  - `max_age = 6`

### 15.2 扫描阈值

- `0.12`
- `0.18`
- `0.24`
- `0.30`

### 15.3 扫描结果

| threshold | block_reuse_ratio | reuse | update |
|---|---:|---:|---:|
| 0.12 | 0.1653 | 242 | 1222 |
| 0.18 | 0.2903 | 425 | 1039 |
| 0.24 | 0.4221 | 618 | 846 |
| 0.30 | 0.5215 | 776 | 712 |

### 15.4 当前观察

这轮结果说明：

1. feature-aware block-skip 的量级已经基本找对了
2. `0.12` 仍然偏保守，矩阵里 update 会很多
3. `0.18` 开始进入可用区间
4. `0.24` 和 `0.30` 更像是值得重点观察的候选

也就是说，当前问题已经不再是：

- “这个 feature-aware 框架通不通”

而是：

- “reuse 的密度应该落在哪个区间更像合理的 skip 逻辑”

### 15.5 当前推荐

下一轮更值得细扫的区间是：

- `0.24 ~ 0.30`

例如可以继续试：

- `0.26`
- `0.28`

因为从矩阵密度角度看，第一轮扫描已经表明：

- 低于 `0.18` 偏保守
- 高于 `0.24` 开始出现比较明显的 reuse 结构

## 16. 2026-05-05 三类 block score 分布审查

已经基于 `can_mh` 的单样本 trace，把三类 block 的 score 分布拆开看过：

- `sa_block`
- `mha_block`
- `ff_block`

### 16.1 核心观察

三类 block 的 score 分布并不一致，而且这个差异在不同 threshold 下都比较稳定：

- `sa_block` 的 score 整体最高
- `mha_block` 居中
- `ff_block` 整体最低

也就是说，当前如果三类 block 共用一个 threshold，会天然导致：

- `ff_block` 更容易 reuse
- `sa_block` 更不容易 reuse

所以从数据上看，**三类 block 不适合长期共用一个完全相同的 threshold**。

### 16.2 以 threshold=0.24 为例

在 `threshold = 0.24` 时，三类 block 的 reuse 比例如下：

- `sa_block`: `0.3607`
- `mha_block`: `0.4426`
- `ff_block`: `0.4631`

对应 score 分布大致为：

- `sa_block`
  - `p25 ≈ 0.195`
  - `p50 ≈ 0.279`
  - `p75 ≈ 0.383`
- `mha_block`
  - `p25 ≈ 0.170`
  - `p50 ≈ 0.246`
  - `p75 ≈ 0.376`
- `ff_block`
  - `p25 ≈ 0.157`
  - `p50 ≈ 0.234`
  - `p75 ≈ 0.307`

### 16.3 排序关系

中位数大致满足：

```text
sa_block > mha_block > ff_block
```

这个排序在目前几轮扫描中都比较稳定。

### 16.4 对 threshold 设计的直接启发

如果以后允许三类 block 使用不同 threshold，那么一个更自然的方向是：

```text
threshold_sa > threshold_mha > threshold_ff
```

原因不是因为 `sa_block` 更“重要”，而是因为它的 score 分布整体更高。

如果继续强行共用同一个 threshold，就会系统性地造成：

- `sa_block` 更新偏多
- `ff_block` 复用偏多

### 16.5 一个合理的第一版异构 threshold 起点

仅从当前分布看，后续如果要尝试分类型 threshold，可以先从下面这种关系开始试：

- `sa_block`: 稍高
- `mha_block`: 中间
- `ff_block`: 稍低

例如第一轮可以试：

- `sa_block = 0.26`
- `mha_block = 0.24`
- `ff_block = 0.22`

这还不是最终答案，但作为第一轮异构 threshold 的起点是有数据依据的。

## 17. block-skip 和 step-skip 的联动问题

当前这版 feature-aware block-skip 仍然有一个根本局限：

- 它把 block reuse 主要看成“当前 block 输入和历史 block 输入像不像”
- 但它没有真正把 `step-skip` 带来的轨迹变化纳入 block 风险判断

这会导致 block-skip 的问题被过度简化。

### 17.1 为什么 step-skip 一定会影响 block-skip

当一个 diffusion step 被 skip 时，发生的事情不是“系统静止了”，而是：

1. 当前 step 没有重新跑模型
2. 直接复用了上一轮 `noise_pred / model_output`
3. scheduler 仍然推进 latent 轨迹

也就是说：

- latent 在变
- denoising 轨迹在往前走
- 但模型内部的信息更新没有同步发生

因此在下一次真正恢复 compute 时：

- block cache 的“陈旧程度”不应该只由局部特征相似度决定
- 还应该由前面累计发生了多少次 step-skip 决定

### 17.2 当前相似度打分为什么不够

单纯的 block 输入相对 L2 drift 只能回答：

- “当前 block 输入和上次参考输入差了多少”

但它回答不了更重要的问题：

- “这个 block 现在的误差风险，是不是因为前面 step-skip 累积而被放大了”
- “当前这个 diffusion 阶段，对 block 误差到底敏不敏感”
- “这一次 compute 是不是刚从连续 skip 之后恢复回来”

所以当前相似度分数更像是：

- 一个局部静态相似性指标

而不是：

- 一个真正反映轨迹误差风险的联动指标

### 17.3 更合理的视角：block risk 不是局部相似度，而是轨迹误差预算

更深一层看，block-skip 应该和 step-skip 共用同一个“误差预算”视角。

block 是否应该更新，应该由下面几类信息共同决定：

1. block 局部变化
   - 当前 block 输入和参考输入差多少
2. 全局 step 状态
   - 上一步是否 step-skip
   - 连续 skip 了几步
   - 当前 step 的 skip 判定 margin 大不大
3. diffusion 阶段
   - 当前 timestep 处在早期、中期还是后期
4. 恢复点风险
   - 当前这一步是不是刚从一串 skipped steps 恢复回来

### 17.4 一个更像样的联动方向

可以把 block 的判断写成：

```text
block_risk = local_block_drift
           + skip_debt_term
           + timestep_stage_term
           + step_uncertainty_term
```

其中：

- `local_block_drift`
  - block 输入的相对变化
- `skip_debt_term`
  - 前面累计 step-skip 的“债务”
- `timestep_stage_term`
  - 当前 diffusion 阶段的敏感性
- `step_uncertainty_term`
  - step-skip 判定本身离阈值有多近

这比单纯做相似度阈值判断更合理，因为它把 block reuse 放回到整条 diffusion 轨迹里理解。

### 17.5 一个很实用的简化版联动实现

在不引入学习器的前提下，最值得先试的不是更复杂的相似度，而是：

#### A. skip debt

维护一个全局 `skip_debt`：

- 每发生一次 step-skip，`skip_debt += 1`
- 每发生一次真实 compute，`skip_debt` 衰减或清零

然后让 block threshold 随 `skip_debt` 动态变化：

- `skip_debt` 越大，block 越保守
- 也就是 effective threshold 越低，越容易触发更新

#### B. resume refresh

如果当前 step 是“连续若干 step-skip 后第一次恢复 compute”：

- 对某些敏感 block 强制更新
- 或者临时降低这些 block 的 threshold

这会比静态相似度更符合轨迹误差传播的直觉。

#### C. step-conditioned threshold

不要给 block 用固定 threshold，而是：

```text
effective_threshold = base_threshold * f(step_state)
```

其中 `f(step_state)` 由下面变量决定：

- `prev_step_skipped`
- `consecutive_step_skips`
- `current_timestep`
- `step_skip_margin`

### 17.6 当前最值得优先尝试的不是更复杂相似度，而是联动机制

所以后面真正值得做的，不是继续在：

- cosine
- L1
- L2
- 混合相似度

这些指标之间来回试太久。

更值得优先尝试的是：

1. block 输入 drift 保留
2. 再显式加入 step-skip 联动变量
3. 先做一个无学习器的 risk score

因为现在的问题更像是：

- 缺少全局轨迹信息

而不只是：

- 局部相似度公式选得不够漂亮

### 17.7 当前推荐的下一步

如果继续往前推，当前最推荐的方向是：

1. 固定 step-skip 不动
2. 保留 block 输入 drift
3. 加入 `skip_debt`
4. 加入“resume step 强制更保守”的机制
5. 再看矩阵和最终分数

也就是说，下一版 block-skip 不应该再只是：

```text
score < threshold ? reuse : update
```

而应该变成：

```text
risk(local_drift, skip_state, timestep_state) < threshold ? reuse : update
```
