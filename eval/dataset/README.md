# T10 评测集说明

`eval.json` 是 T10 的评测集：一个虚构用户的笔记库（Python / 机器学习 / 求职 / 生活）
+ 38 条样本。语法由 `load_dataset`（`eval/runner.py`）校验，语义由
`test_dataset_integrity`（`tests/test_eval.py`）守住。

## 文件结构

```
eval/dataset/
├── eval.json     # 评测集：docs（要 ingest 的文档）+ samples（样本）
└── docs/         # 11 篇 Markdown 笔记，源文件
```

## 字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `version` | 是 | 评测集版本，进 `EvalResult.dataset_version` 与报告快照 |
| `docs` | 是 | 相对 `eval/dataset/docs/` 的文件名列表；runner 启动时全量 ingest |
| `samples[].id` | 是 | 样本唯一 id（`eval-NNN`），进结果里的 `SampleResult.id` |
| `samples[].category` | 是 | `single-hop` / `multi-hop` / `temporal` / `preference` / `adversarial` |
| `samples[].query` | 是 | 用户提问 |
| `samples[].expected_chunks` | 否 | 期望检索到的**文档标题**（不是 chunk_id，见下）；空表示该样本不考核检索 |
| `samples[].expected_answer_contains` | 否 | 回答必须覆盖的关键词，命中比例即关键词覆盖率；空视为无约束（不惩罚） |
| `samples[].expected_answer_excludes` | 否 | 回答**不得出现**的关键词；出现即该样本覆盖率 0（对抗样本的判别力来源，见下） |
| `samples[].expected_memories` | 否 | 期望召回的记忆条目；空表示不考核记忆（聚合时跳过而不是算 0 分） |
| `samples[].seed_memories` | 否 | 评测前预置进 `memories` 表的记忆条目 |
| `samples[].sessions` | 否 | 评测前写入该会话 `messages` 表的「历史」问答 |
| `samples[].notes` | 否 | 人读的备注，不参与评分 |

### 为什么 `expected_chunks` 写文档标题

`chunks.id` 是数据库自增主键，重新 ingest 一次就整体平移，写进评测集会随运行次数失效。
文档标题是稳定的业务标识。runner 把每个检索到的 chunk 映射回标题再比对
（`eval/runner.py` 的 `_install_observers`）。

因此**同一篇文档的标题必须唯一**，改标题等于改评测集。

### `expected_answer_excludes` 解决什么问题

对抗样本的期望是「明确说明没有」，但只写 `expected_answer_contains: ["没有"]` 没有判别力：
胡编一段内容再补一句「（笔记里）没有」同样满分。禁词给出**只有编造才可能命中**的说法
（如 Rust 的「借用检查器」、K8s Ingress 的「nginx」），把「正确拒答」和「编造 + 搪塞」
区分开。加禁词时挑「回答里必然不会出现的具体术语」，太泛的词（如「知识点」）会误伤。

## 样本分布

| 类别 | 占比目标 | 当前 |
|------|---------|------|
| single-hop | 30% | 12 |
| multi-hop | 25% | 10 |
| temporal | 20% | 7 |
| preference | 15% | 5 |
| adversarial | 10% | 4 |

`test_dataset_integrity` 断言实际占比与目标的偏差 ≤ 0.08。

## 已知局限（读结论前必看）

- **每篇文档单 chunk，Hit@k 的随机基线偏高。** `chunk_size=500`、11 篇文档都很短，
  实测每篇只切出 1 个 chunk，库内共 11 个 chunk。k=8 时**随机**排序的 Hit@k 就有
  8/11 ≈ 0.73，MRR 也在 0.5 上下量级。所以绝对数值不能当作「检索能力」的度量——
  **结论以组间相对对比为准**（同一 k、同一库规模下的消融差异），不要拿 hit@k=0.8
  去说「检索准确率 80%」。要降低基线就扩文档数（每篇仍单 chunk 也能把基线压到 k/N），
  或把长文拆成多 chunk——前者更省事，但会改变全部样本的期望标注。
- **样本量 38，不足以做统计显著性检验。** 8 组消融 × 38 样本的差异里，单样本翻转
  就能让 hit@k 动 2~3 个百分点。报告只给点估计，不下「显著提升」的结论。
- **长会话样本只有 3 条（eval-036/037/038，14 条历史）。** `context_compaction_enabled`
  在历史条数 > `context_compaction_threshold`（默认 10）时才触发，其他样本最多 2 条
  历史，永远压不到——没有这几条时消融的「压缩」轴的两组（A/C、D/G）输出逐字节相同，
  等于没测。加样本时保持「> 12 条历史」这个形态。
- **temporal 类靠文档正文里的日期文字**（如 FastAPI 笔记里的「记录时间：2026-09-14」）
  由模型自行推断相对时间，不是真正的结构化时间检索。样本里的「上周」「第三周」以
  2026-09 为基准。
