# methods-ledger · 方法与实验章节写作（写得全 · 写得准 · 做得着 · 说得住）

面向「写作09｜方法与实验章节写作」：**文字由模型写，每个数字由脚本对着来源核对**。脚本从配置文件、代码、日志、结果表、依赖文件和实验笔记里读出参数、数据集、环境、命令和结果，每个值都带着文件和行号；来源之间互相矛盾时列成冲突、不替作者选择；模型写出的章节逐数字对照这些事实，找出凭空补出的数值、与来源不一致的取值、训练与评测条件串台、遗漏的要素和空泛的限制；材料里没有的信息用 `[[CONFIRM: …]]` 明确标出，汇成交作者确认的问题清单。

> 文中数字来自开发自测（虚构的实验项目 + 6 个公开 PyTorch 示例脚本 + 27 个公开 YAML + 10 个公开 pyproject.toml），**不是平台成绩**；没有跑过 DeepSeek 官方 Harness，也没有做过「不加载 Skill 的基础 Agent」对照，因此不宣称相对基线的提升。

## 对应题面的四个问题
| 题面问题 | 做法 | 自查手段 |
|---|---|---|
| 写得全 | 18 个要素清单（随机种子、重复次数与统计、训练长度、批大小、优化器、学习率与调度、硬件、软件版本、数据集名称 / 版本 / 规模 / 划分 / 预处理、指标、代码版本、命令、数据获取）：材料里有而正文没写 → `element_omitted`；材料里没有且正文没标 → `gap_not_marked` | `check` 的要素表；`questions.md` |
| 写得准 | 正文每个数字、每个“参数名 + 取值”都要对上事实：对不上 → `value_mismatch`（列出来源给的值）或 `unsourced_number`；框架 / CUDA / Python 版本、GPU 型号、优化器名、数据集版本同样核对；运行日志 / 命令行能判定的取值自动采信，其余冲突保持开放 | `source_map.csv`（每个数字→来源）；`parameters.md` |
| 做得着 | 复现清单：命令从脚本和日志里原样取出（`commands.md`），代码版本、环境文件、数据获取缺失就是待确认项，不猜 | `repro.*` 要素；`questions.md` |
| 说得住 | `[[CONFIRM: …]]` 把缺口留在正文里并进入 `questions.md` 与 `result.json`；限制必须是失败场景（`generic_limitation` 检出“可能不能泛化”式套话）；“同等预算 / 公平比较”没有来源就告警 | `review_checklist.md`、`result.json` 的 `boundaries` |

## 目录
```
SKILL.md                         Agent 入口：何时用、八个步骤、冲突处理块、写作规则、异常处理
README.md                        本文件
scripts/mw.py                    命令行：run / show / skeleton / ingest / check / build
scripts/extract.py               读取层：配置、Python（AST）、笔记本、脚本命令、日志、CSV、依赖文件、中英文散文 → 带文件行号的事实
scripts/facts.py                 冲突判定（运行证据优先）、参数 / 数据集 / 环境表、要素清单与提问模板
scripts/mini.py                  标准库写的 YAML / TOML / INI / .env 读取器
scripts/docxw.py                 标准库写 .docx；scripts/nlib.py, segment.py, parse_pdf.py 读取 docx / pdf / 分句
references/                      读取与匹配规则、全部检查项与实测、写作指南与提问模板
examples/                        虚构实验项目 + 有缺陷的初稿与修订稿的真实运行 + 中文案例
GITHUB_URL.txt                   作品的 GitHub 公开链接
dev/ （仅仓库）                  36 个回归测试、抽取 / 核对评测、公开样例数据、示例生成脚本
```

## 依赖与安装
Python 3.9+，只用标准库（已在系统 Python 3.9.6 上跑通）；`.pdf` 笔记需要环境里有 `pymupdf`、`pdfplumber` 或 `pypdf` 之一。无需联网。输入：`/input/case.json` + `task.md`（语言、字数从中读取）及其所在文件夹（代码、配置、日志、表格、笔记，最多 5 层），`case.json` 里的长文本字段和 `settings / config / params / 参数` 字典，或 `--src DIR`。

## 使用步骤
```bash
python3 scripts/mw.py run --case /input/case.json --out out        # 简报 out/brief.md：参数、冲突、环境、结果、命令、缺失项
python3 scripts/mw.py ingest out/.work decisions.txt               # resolve（要原因）/ confirm（要逐字引文）
python3 scripts/mw.py skeleton out/.work                           # 可选：以事实填好、缺口标 [[CONFIRM]] 的提纲
python3 scripts/mw.py check out/.work draft.md                     # 逐数字核对 + 要素 + 限制
python3 scripts/mw.py build out/.work draft.md --out out           # 章节 + 表 + 问题清单 + result.json
```

## 输出
`section.md / .docx / .tex`、`parameters.csv / .md`（参数、取值、阶段、状态、来源）、`datasets_environment.csv`、`commands.md`、`questions.md`（交作者确认）、`source_map.csv / .json`（每个数字对应的事实）、`review_checklist.md`、`result.json`（按 `output_contract` 的字段填充，推导不出的置 `null` 并列在 `_meta.unfilled_fields`；有错误或未决冲突时 `status` 为 `partial`）。

## 实测（开发自测，不是平台成绩）
| 项 | 结果 |
|---|---|
| 回归测试 | `python3 dev/test_all.py`：36 个通过：YAML / TOML 读取、argparse / 调用 / 常量 / dataclass、日志与散文模式、冲突判定、植入缺陷（改参数、改版本、GPU、数据集版本、凭空数字、训练 / 评测串台、占位符、套话限制、公平比较）、中文稿、`resolve` / `confirm` 的拒绝与接受、已有初稿不当证据、无材料与损坏文件不崩溃 |
| 数字核对（虚构项目，22 条参数陈述，英 / 中） | 忠实陈述 22 / 22 无错误；改动取值后 21 / 22 被标出（漏的一条是 64 的一半恰为训练值 32）；6 条材料里完全没有的参数陈述 6 / 6 被标出。规则是看着这个固定项目调出来的，数字偏乐观 |
| 抽取（公开数据） | 6 个 PyTorch 官方示例脚本的 argparse 默认值 65 / 65 取回且取值相同；YAML 读取器对 PyYAML：27 个公开文件（detectron2、Kubernetes、GitHub Actions、docker-compose、yolov5 ……）标量叶子 27 / 27 相同；TOML 读取器对 tomli：10 个公开 pyproject.toml 8 / 10 相同（差异在字面量字符串的转义） |
| 示例 | `examples/`：第一稿 16 个错误 → 修订稿 0 个错误（3 个未决冲突与 8 项待作者确认如实保留，`status: partial`）；中文案例 |

## 限制（如实写在这里）
- 值是从文件的**文字**里读的：运行时才算出的值、环境变量设置的值、文件里看不到的代码分支读不到；日志 / 命令行优先于配置、配置优先于代码默认值，这是约定而不是真相，所以冲突会列出来、`resolve` 要给理由。
- 要素检查是词汇层面的（正文里出现相应说法），不判断说法是否充分；数字核对不理解语义（“学习率衰减 10 倍”这类变换不会被追溯）。
- 不核对结果数字是否“正确”，只核对它与日志 / 结果表一致；不评价方法本身。
- 日志格式千差万别：只识别常见模式（`key: value`、`Namespace(...)`、`Epoch i/N`、指标行、环境横幅）；TOML 读取器对少数转义写法不完全一致。
- 没有 Word / LibreOffice 渲染验证 `.docx`，只验证了 XML 可解析。

## 方法说明
1. **读取层**：每种文件一个读取器，值只来自文件里写着的内容，带文件与行号；名字映射到约 25 个规范参数，阶段（训练 / 验证 / 测试）来自键路径、调用、文件名（`eval.py`）或句子中的子句（中英文）。
2. **冲突**：运行证据（日志、命令行）一致即采信；配置与代码里写死的值不一致、笔记与配置不一致等列为冲突，作者用 `resolve`（给原因）决定；决定后表格与核对都用决定值。
3. **核对**：数字规范化（`1e-3` = `0.001` = `1×10^-3`，`87.1%` = `0.871`，`8:1:1`）后与事实比对，前后文名出参数就必须等于该参数的事实；常见小数值只在句子与事实共享词时才算对上，避免巧合。
4. **写作与核对分离**：脚本从不替你写句子，也不补值；`skeleton` 只用事实填，缺口写成 `[[CONFIRM]]`；`build` 只做清理、表格、转换，并把未决事项写进 `questions.md` 与 `result.json`。

## 复现
`python3 dev/test_all.py`；`python3 dev/eval_check.py`；`python3 dev/eval_extract.py`（需 `pip install --target dev/_deps pyyaml tomli`）；`python3 dev/make_examples.py`。


## 示例文件的后缀
平台只接受部分文件类型作为 Skill 资源，所以示例里这些文件的名字后面加了 `.txt`：`examples/project/run.sh`、`examples/project/logs/train.log`。去掉 `.txt` 就是脚本实际读写的文件名；脚本本身的输出文件名没有变。
