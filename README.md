# A-Share Alpha Discovery

一个使用真实 A 股历史数据、严格记录失败路径并进行资产隔离复现的量化研究项目。

最终产物是 **Supply-Aware Defensive Attention（SADA）**：将五个点时技术/交易行为
信号等权合成，再用尚未被成交量吸收的流通股供给库存施加单边风险惩罚。SADA 是本仓库
的新工程复合公式；MAX、低换手、日内反转、低波动和短期反转本身均有既有文献，因此本
项目不声称发现了全新的基础异象或因果机制。

## 核心结果

| 阶段 | 股票 | 月数 | 平均 Rank IC | HAC p / BH q | 月均净主动收益 | 年化月度 IR |
|---|---:|---:|---:|---:|---:|---:|
| A：发现 2020–2022 | 2,149 个历史并集 | 36 | 0.1207 | BH q = 6.17e-14（累计 45 次检验） | — | — |
| A：内部验证 2023–2024 | 同上 | 24 | 0.1415 | — | 0.6826% | 1.085 |
| B：资产隔离复现 2020–2026.05 | 2,124 个历史并集 | 77 | 0.1222 | HAC p = 5.72e-22 | 0.5373% | 0.872 |

B 与 A 的股票代码交集为 **0**。组合为月度 top-100 等权，持仓因子分位不低于 60% 时
保留，交易成本为单边 20bp。未来收益使用下一市场交易日开盘到第 21 个市场交易日开盘；
最多等待 5 个交易日，无法进场则持有现金，无法按期开盘退出则按登记规则使用最后可得收盘价。

重要限制：B 对 SADA 公式是在冻结后首次评估，但这套 B 股票池曾在旧项目中用于其他因子
研究。因此结果应称为“预注册公式的资产隔离历史复现”，不是全球完全未触碰的纯净 holdout。
此外，2026 年 1–5 月的 IC 为 -0.0030，净主动收益为 -1.6747%/月，提示近期失效风险。

机器可读摘要位于 [`results/`](results/)，完整审计见
[`docs/statistical_audit.md`](docs/statistical_audit.md)。

## 因子公式

每天在当期合格股票横截面内，对下列分量做 3-MAD 缩尾、z-score，并截断到 `[-5, 5]`：

1. `anti_max_return_20`：过去 20 日最大市场调整收益的负值；
2. `low_turnover_20`：过去 20 日平均换手率的负对数；
3. `negative_intraday_mean_20`：过去 20 日日内市场调整收益均值的负值；
4. `low_volatility_60`：过去 60 日收益波动率的负值；
5. `reversal_20`：过去 20 日收益率的负值。

核心分数是五个 z-score 的等权平均。动态流通股库存按下式递推：

```text
supply_t    = max(float_shares_t - float_shares_{t-1}, 0) / float_shares_t
inventory_t = (inventory_{t-1} + supply_t) * exp(-turnover_t / 0.25)
supply_days = -log(1 + inventory_t / mean_turnover_20)

SADA = mean(z_antiMAX, z_lowTurn, z_negIntra, z_lowVol, z_reversal)
       - 0.25 * max(-z_supply_days, 0)
```

月末 SADA 再对点时流通市值对数、其中心化平方项和申万一级行业哑变量做 OLS 残差化，
最后标准化。B 中规模与行业残余暴露的最大绝对值为 `2.14e-14`。

## 研究设计

- 真实数据：腾讯后复权 OHLC；通达信成交量、历史流通股本与交易状态；申万历史行业。
- 点时处理：滚动窗口只向后看，行业使用 `start_date` 向后 as-of 对齐。
- 样本：每月 500 只高流动性非指数股票；排除当月 CSI 300/500；A/B 用稳定资产哈希切分。
- 研究期：A 的 2020–2022 为发现、2023–2024 为内部验证。
- 多重检验：V1–V6 与基准诊断共 45 个方向性检验，统一纳入 BH-FDR。
- 失败保留：V1–V5 的供给、缺口吸收、尾部集中和非线性交互等失败结果全部写入 `docs/`。
- 复现：A 选择阶段和 B 复现阶段分别完整重跑，全部 10 个输出文件 SHA-256 字节一致。
- 测试：18 个单元/回归测试，包含中性化正交、共享日历执行、缺失成交处理和前视不变性。

成本压力测试：

| 单边成本 | B 月均净主动收益 | 年化月度 IR |
|---:|---:|---:|
| 0bp | 0.7604% | 1.235 |
| 10bp | 0.6488% | 1.053 |
| 20bp | 0.5373% | 0.872 |
| 30bp | 0.4258% | 0.691 |
| 50bp | 0.2028% | 0.329 |

## 复现方法

Python 3.10+：

```powershell
python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
python -m pytest -q
```

先运行 A 研究门：

```powershell
python scripts/run_discovery_v6.py `
  --data path/to/liquid_bucket_a_structural_2020_2026.parquet `
  --hfq-cache path/to/tencent_hfq `
  --industry path/to/sw_industry_history.parquet `
  --prior-summary reports/discovery_v1/candidate_summary.csv `
  --prior-summary reports/discovery_v2/candidate_summary.csv `
  --prior-summary reports/discovery_v3/candidate_summary.csv `
  --prior-summary reports/discovery_v4/candidate_summary.csv `
  --prior-summary reports/baseline_diagnostics/candidate_summary.csv `
  --prior-summary reports/discovery_v5/candidate_summary.csv
```

仅当 A 的 `selected_candidate` 为 SADA 时，运行一次 B 复现：

```powershell
python scripts/run_confirmation_v6.py `
  --research-data path/to/liquid_bucket_a_structural_2020_2026.parquet `
  --confirmation-data path/to/liquid_bucket_b_structural_2020_2026.parquet `
  --hfq-cache path/to/tencent_hfq `
  --industry path/to/sw_industry_history.parquet
```

原始行情与生成的逐月报告因许可和体积不进入 Git。材料指纹、公开摘要和环境版本记录在
[`results/material_passport.yaml`](results/material_passport.yaml)。

## 目录

```text
configs/    冻结的 V1–V6 研究与确认门槛
docs/       协议、每轮负结果、统计谬误与限制审计
results/    可提交的摘要、成本压力与复现哈希
scripts/    发现、基准校准和一次性确认入口
src/        数据对齐、因子、组合、统计检验和确认逻辑
tests/      单元、边界和前视不变性测试
```

## 既有研究与新颖性边界

反 MAX 在中国市场已有直接证据，例如 [Nartea, Kong & Wu (2017)](https://doi.org/10.1016/j.jbankfin.2016.12.008)，
涨跌停环境下的修正 MAX 见 [Yao et al. (2021)](https://doi.org/10.1016/j.iref.2021.01.014)，
隔夜/日内彩票效应分解见 [Gu, Hu & Xiong (2025)](https://doi.org/10.1111/acfi.13354)，
中国流通股供给制度背景见 [Fang et al. (2017)](https://doi.org/10.1016/j.jbankfin.2017.08.012)。

SADA 的可主张贡献是：一个完全固定、规模与行业中性的工程公式；一个把动态供给吸收状态
作为单边风险约束的实现；以及带累计多重检验、失败日志、资产隔离复现和字节级复现证据的
完整研究流程。它不证明因果、实盘盈利或全球学术原创性。
