# A-Share Alpha Discovery

> Latest audit (2026-07-25): 131 directional tests; no factor has passed every
> frozen gate. The new official margin-financing study is documented in
> [`docs/margin_orderbook_research_v48_v56.md`](docs/margin_orderbook_research_v48_v56.md).
> `hgb_tail_v53` independently replicated rank IC in B (`0.02627`, p=`0.00957`)
> but failed cost-adjusted return and IR, so it is not presented as an effective
> factor. Newly built zero-overlap buckets C/D remain unopened.

一个使用真实 A 股历史数据、点时市值与行业中性化、累计多重检验和封存确认集的因子挖掘审计项目。

## 当前结论

**截至 2026-07-24，本项目没有发现一个可以诚实称为“新的、通过全部预注册门槛的有效因子”。**

项目累计记录 108 个方向性候选检验。V19 曾通过研究样本，但在一次性资产隔离确认集 D 上方向失效，因此正式作废。V20–V40 在开发样本 E 上继续研究，但没有候选通过完整研究门槛。V41 的 E 适应性诊断与 F 独立确认均失败。V42–V44 把已消耗的 E+F 用作构造数据但均未通过。V45 首次完整通过 E+F 构造门槛，却在公式未见、资产完全不重叠的 G 上确认失败。V46 把 G 降级为构造数据后仍未通过；V47 虽通过 E+F+G 构造，却在新封存的 172 只资产 H 上显著反向。因此截至当前仍没有可以诚实称为有效的新因子。

这是刻意保留的负结果，不是未完成的回测。仓库拒绝以下做法：

- 看见负号后翻转因子方向；
- 反复打开确认集调窗口、权重或阈值；
- 忽略先前失败以缩小多重检验家族；
- 用市值、行业、低换手或低波动暴露冒充新因子；
- 把“接近显著”写成“已验证有效”。

## 最接近但仍失败的候选

| 候选 | 发现期 | 时间外验证 | 成本后表现 | 失败原因 |
|---|---|---|---|---|
| `fsid_120` | IC 0.01745，BH q 0.195 | IC 0.01684，24 个月、两年为正 | +0.1433%/月，IR 0.78 | 累计多重检验未通过 |
| `fsid_100` | IC 0.02826，BH q 0.0487 | IC 0.00866，仅 20 个合格月 | +0.0974%/月，IR 0.55 | 验证 IC、月份数和逐年门槛未通过 |
| `nsim_v30` | 训练期 IC 0.1532 | IC 0.01647，HAC 单侧 p 0.05352 | +0.2043%/月，IR 0.51 | p 高于冻结上限 0.05 |

`fsid`（Float-Supply Impact Decay）是本轮最有研究价值的原创精确构造：在流通股本增加事件完整经过 20 个市场交易日后，比较第 1–5 日与第 6–20 日的市场残差收益/累计换手，衡量新增流通供给的单位换手价格冲击是否衰减。它有经济与时间外信号，但尚未获得足够严格的统计确认。

## V41 双门槛审计

`scfc_v41` 先对两个冻结组件分别做横截面中性化和月内百分位排序，再等权合成：

```text
scfc_v41 = 0.5 × rank(neutralized nsim_v30)
          + 0.5 × rank(neutralized cfma_v40)
```

最终分数再次剔除 14 个价量主效应、15 个财务主效应、流通市值一次/二次项和申万一级行业。最大绝对残余暴露为 E `1.13e-14`、F `2.71e-14`，因此失败不能归因于未处理中小市值或行业暴露。

| 证据 | 角色 | 月数 | 平均 Rank IC | HAC 单侧 p | 扣费后主动收益/月 | IR | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| E，2023–2025 | 适应性 post-fit 诊断，不是样本外 | 36 | 0.00903 | 0.1941 | −0.1421% | −0.390 | 失败 |
| F，2020–2025 | 一次性资产隔离确认 | 70 | 0.01028 | 0.0299 | −0.2622% | −0.624 | 失败 |

E 还因 2025 年 IC `−0.01129` 和三年一致性失败；F 虽有统计显著的正平均 IC，但低于冻结的 `0.015` 门槛，2024 年 IC 为负且成本后组合明显亏损。联合有效性为 `False AND False = False`。

V42 `tafs_v42` 是第 95 个候选：E+F 的 2023–2025 构造期 IC `0.00470`、HAC p `0.1059`，2024 年 IC 为负；其 50 股组合扣费后月均 `+0.3811%`、IR `0.644`，但联合门槛不允许用组合收益替代 IC 和年度稳定性，所以仍判失败，G 未打开。

V43 `rrsm_v43` 是第 96 个候选：构造期 IC `0.01820`、HAC p `0.00010`，2023–2025 三年 IC 均为正；但 2025 年 top-50 月均 `−1.5686%`，全期扣费后月均 `−0.2936%`、IR `−0.442`，仍未通过联合门槛。

V44 是第 97–99 个候选。最接近的 `rrsm60_tafs40_v44` 在构造期取得 IC `0.01477`、扣费后月均 `+0.1583%`，但分别低于 `0.015` 的 IC 门槛和 `0.30` 的 IR 门槛（实际 IR `0.250`）；另外两条组合虽然 IC 合格，但成本后收益为负。三条均失败，G 未打开。

V45 是第 100–102 个候选。被预注册规则选中的 `pcs60_25_15_v45` 在 E+F 构造期取得 IC `0.01757`、HAC p `0.000074`、扣费后月均 `+0.2687%`、IR `0.515`，完整通过；但 G 一次性确认只有 IC `0.01051`、HAC p `0.0933`，且 2023 年 IC 为负。G 的扣费后月均 `+0.2355%` 和 IR `0.804` 不能覆盖失败的统计门槛，因此不声明有效因子。

V46–V47 是第 103–108 个候选。V46 的稳健聚合全部在构造期失败。V47 的 `wpen10_v47` 在 E+F+G 上 IC `0.01755`、扣费后月均 `+0.2084%`、IR `0.371`，但 H 一次性确认 IC `−0.02576`，2023/2024 均为负，扣费后月均 `−0.2299%`、IR `−1.030`，正式作废。

## 研究流程

```text
候选机制与先验检索
        ↓
冻结公式、方向、窗口、控制变量和门槛并提交 Git
        ↓
开发期 2020–2022 + 明确标记证据角色的后续诊断
        ↓
累计 Benjamini–Hochberg + 逐年 IC + 20bp 成本 + IR
        ↓
仅按冻结权限一次性打开资产隔离确认集，失败后永久禁止调参复用
```

每个月末候选分数均剔除：

- 点时流通市值对数及其中心化平方项；
- 点时申万一级行业哑变量；
- 候选注册的传统主效应和机制专属控制项。

未来收益为下一个市场交易日开盘到第 21 个市场交易日开盘；最多允许 5 个市场交易日的成交延迟。组合为缓冲 top-100 等权多头，单边成本 20bp。

## 数据

- 行情：腾讯后复权 OHLC；
- 成交结构：本地通达信实际成交量、成交额与 `gbbq` 历史流通股本；
- 行业：申万一级历史分类，按 `start_date` 向后 as-of 对齐；
- 开发 E：2,149 只股票，3,155,290 行，SHA-256 `6a779e87...cec13`；
- 封存 F：2,124 只股票，与 E 代码交集为 0，SHA-256 `a026ac04...b2f9`。

重要限制：E/F 曾被旧仓库用于另一条 SADA 工程公式，因此它们不是全球完全未触碰的样本。E 的 2023–2025 区间已被适应性查看，只称为内部诊断，不称为纯净样本外。F 与 E 资产不重叠，但共享日历时间，也不是未来时间样本外。历史 ST 状态不完整，当前结构表把 `is_st` 置为 0；这一限制禁止项目宣称全市场无偏或可直接实盘。V32–V41 使用 2026-07-23 下载的通达信专业财务包并按公告日门控，但无法排除后续更正或重述。

## 目录

```text
configs/    每版冻结协议与看结果前否决记录
docs/       数据契约、研究协议、完整失败账本与统计审计
results/    可提交的统一候选表、状态清单与材料护照
scripts/    数据构建、单因子发现、学习型因子和一次性确认入口
src/        因子、模型、点时对齐、中性化、组合与统计实现
tests/      前视不变性、执行规则、统计与协议常量测试
```

## 复现

Python 3.10+：

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m pytest -q
```

单公式版本通过统一入口运行，例如 V29：

```powershell
python scripts/run_adaptive_discovery.py `
  --version 29 `
  --data path/to/liquid_bucket_a_structural_2020_2026.parquet `
  --hfq-cache path/to/tencent_hfq `
  --industry path/to/sw_industry_history.parquet `
  --prior-summary path/to/each_prior_candidate_summary.csv `
  --protocol configs/discovery_v29.yaml `
  --output reports/discovery_v29
```

学习型候选分别使用 `scripts/run_model_discovery_v30.py` 和 `scripts/run_model_discovery_v31.py`。命令会校验候选注册表、既往检验数量和冻结超参数；模型验证失败时不会生成确认权限。

## 可核查证据

- [完整研究账本](docs/research_ledger_v7_v47.md)
- [V32–V41 专业财务数据与因子审计](docs/financial_factors_v32_v41.md)
- [统计与偏差审计](docs/statistical_audit.md)
- [108 个候选统一表](results/all_candidate_tests.csv)
- [V41 E/F 联合汇总](results/v41_joint_summary.csv)
- [V41 逐年结果](results/v41_yearly_results.csv)
- [V41 完整审计与哈希](results/v41_audit.yaml)
- [V42 构造汇总](results/v42_construction_summary.csv)
- [V42 审计与哈希](results/v42_audit.yaml)
- [V43 构造汇总](results/v43_construction_summary.csv)
- [V43 审计与哈希](results/v43_audit.yaml)
- [V44 构造汇总](results/v44_construction_summary.csv)
- [V44 审计与哈希](results/v44_audit.yaml)
- [V45 构造与 G 确认汇总](results/v45_joint_summary.csv)
- [V45 完整审计与哈希](results/v45_audit.yaml)
- [V46 构造汇总与审计](results/v46_construction_summary.csv)
- [V47 构造与 H 确认汇总](results/v47_joint_summary.csv)
- [V47 完整审计与哈希](results/v47_audit.yaml)
- [最终研究状态](results/research_status.yaml)
- [材料护照](results/material_passport.yaml)

本仓库展示的是可信的量化研究流程，而不是保证盈利的策略。历史相关性不证明因果，也不保证未来收益。
