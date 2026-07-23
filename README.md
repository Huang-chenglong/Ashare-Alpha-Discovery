# A-Share Alpha Discovery

一个使用真实 A 股历史数据、点时市值与行业中性化、累计多重检验和封存确认集的因子挖掘审计项目。

## 当前结论

**截至 2026-07-23，本项目没有发现一个可以诚实称为“新的、通过全部预注册门槛的有效因子”。**

项目累计记录 87 个方向性候选检验。V19 曾通过研究样本，但在一次性资产隔离确认集 D 上方向失效，因此正式作废。V20–V34 在公式新鲜的开发样本 E 上继续研究；没有候选通过完整研究门槛，所以确认集 F 始终未被打开。

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

## 研究流程

```text
候选机制与先验检索
        ↓
冻结公式、方向、窗口、控制变量和门槛并提交 Git
        ↓
开发期 2020–2022 + 时间外验证 2023–2024
        ↓
累计 Benjamini–Hochberg + 逐年 IC + 20bp 成本 + IR
        ↓
仅全部通过时，才允许一次性打开资产隔离确认集
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

重要限制：E/F 曾被旧仓库用于另一条 SADA 工程公式，因此它们不是全球完全未触碰的样本；V20–V34 只满足“精确公式未在该桶上评估”的新鲜度。历史 ST 状态不完整，当前结构表把 `is_st` 置为 0；这一限制禁止项目宣称全市场无偏或可直接实盘。V32–V34 使用 2026-07-23 下载的通达信专业财务包并按公告日门控，但无法排除后续更正或重述。

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

- [完整研究账本](docs/research_ledger_v7_v34.md)
- [V32–V34 专业财务数据与因子审计](docs/financial_factors_v32_v34.md)
- [统计与偏差审计](docs/statistical_audit.md)
- [87 个候选统一表](results/all_candidate_tests.csv)
- [最终研究状态](results/research_status.yaml)
- [材料护照](results/material_passport.yaml)

本仓库展示的是可信的量化研究流程，而不是保证盈利的策略。历史相关性不证明因果，也不保证未来收益。
