# Pipeline Agent 实现与问题回顾

本文档总结 SWE-agent **多模型流水线（Planning → Coding → Verification）** 的已实现内容、运行中遇到的问题、已知限制与后续建议。

---

## 1. 目标与设计

在单条 SWE-bench 实例上串联：

1. **Phase 1 — Planning（大模型）**：在容器内检索相关代码、分析问题，将计划写入 `/root/plan.md` 后 `submit` 结束。
2. **Phase 2 — Coding（小模型）**：在同一容器、同一仓库状态下，根据注入的 `<plan>`（及可选的验证反馈）修改代码并 `submit` 产出 patch。
3. **Phase 3 — Verification（大模型）**：对 patch 打分（0–10）并生成文字反馈；未达 `accept_score` 则 **软重置仓库**（`git` 回到 base，不重启容器）、将反馈注入下一轮 Coding。

与原生 `RetryAgent` 的区别：阶段间**不** `hard_reset()`，可传递 plan；验证失败可把意见回灌给小模型。

---

## 2. 已实现内容

### 2.1 代码与注册

| 位置 | 说明 |
|------|------|
| [`sweagent/agent/agents.py`](sweagent/agent/agents.py) | `VerificationConfig`、`PipelineAgentConfig`；`AgentConfig` union 扩展；`get_agent_from_config()` 中 `type: pipeline` 分支 |
| [`sweagent/agent/extra/pipeline_agent.py`](sweagent/agent/extra/pipeline_agent.py) | `PipelineAgent`：`run()` / `setup()` / `add_hook()`；三阶段循环、软重置、合并 trajectory、`AgentRunResult` |
| [`sweagent/run/batch_instances.py`](sweagent/run/batch_instances.py) | `SWEBenchInstances` 扩展：支持 `reference_repos`、`reference_repo_root`、额外 `post_startup_commands` 注入 |
| [`config/pipeline_qwen.yaml`](config/pipeline_qwen.yaml) | 示例：Qwen3.5-27B（:30001）+ Qwen3.5-9B（:30000），SGLang OpenAI 兼容 API |
| [`scripts/test_pipeline_e2e.sh`](scripts/test_pipeline_e2e.sh) | Smoke / single / batch；无 `SWE_BENCH_API_KEY` 时自动 `evaluate=False`；可选 `--small-url` / `--large-url` |
| [`scripts/inspect_pipeline_traj.py`](scripts/inspect_pipeline_traj.py) | 解析合并后的 `.traj`（含 `pipeline_attempts`），便于排错 |

### 2.2 与 SWE-bench 的衔接

- 输出仍为 `AgentRunResult`，`save_predictions()` 读取 `info["submission"]`；Pipeline 在返回前将 **最佳 patch** 写入顶层 `info`。
- `run-batch` + `SWEBenchInstances` 无需改代码即可跑；云端评测依赖 `sb-cli` 与 API Key。

### 2.3 实现细节要点

- **计划读取**：优先读 `/root/plan.md`，失败则从 planning trajectory 中取较长 `thought` 兜底。
- **软重置**：调用 `repo.get_reset_commands()`，保留容器与 `/root/plan.md`（在仓库外则不受 reset 影响）。
- **验证模型统计**：验证阶段复用同一 `AbstractModel` 实例，`stats` 为累计值；已实现 **按轮次增量** 合并到 `_total_stats`（避免重复累加）。
- **外部参考仓库注入**：支持在 SWE-bench 实例启动后自动 clone/链接参考仓库到 `/root/reference_repos`，供 Planning 阶段读取。
- **参考仓库命名去冲突**：目录名使用 `basename + sha1短哈希`，避免同名仓库互相覆盖。
- **本地路径链接稳定性**：本地 reference 路径注入前会先清理目标目录，再创建符号链接，避免旧目录残留。

---

## 3. 配置说明（SGLang / 本地模型）

### 3.1 模型 ID 与 `api_base`

- SGLang 返回的 `id` 形如 `/models/Qwen/Qwen3.5-9B/` 时，litellm 侧使用：`name: openai//models/Qwen/Qwen3.5-9B/`（双斜杠为转义后的路径形式）。
- `api_base` 指向各服务根，例如 `http://localhost:30000/v1`、`http://localhost:30001/v1`。
- 无鉴权时可用 `api_key: EMPTY` 等非空占位。

### 3.2 费用与 litellm

- 对 **未在 litellm 计价表注册** 的模型，`completion_cost` 会抛错；若 `per_instance_cost_limit > 0` 或 `total_cost_limit > 0`，SWE-agent 会 **直接报错退出**（`ModelConfigurationError`）。
- **本地 / 自托管模型**：应将上述两项及 pipeline 的 `cost_limit` 设为 **0**，用 `per_instance_call_limit` 控制调用次数。
- `pipeline_qwen.yaml` 中已按此配置。

### 3.3 解析器

- Planning 仅 `registry` + `diff_state`，无 `edit_anthropic`、`review_on_submit_m`，避免「无 diff 的 submit 审查」干扰。
- Planning / Coding 使用 `thought_action` 解析，对 SGLang 上部分模型比 `function_calling` 更稳；Coding 仍带 `edit_anthropic` + `review_on_submit_m` + `diff_state`。

---

## 4. 运行与测试命令

```bash
# 仅校验 import、YAML、SGLang 连通性
bash scripts/test_pipeline_e2e.sh --smoke

# 单条 SWE-bench Lite dev（无 API Key 则跳过云端评测）
bash scripts/test_pipeline_e2e.sh --single

# 带外部参考仓库（可重复传入）
bash scripts/test_pipeline_e2e.sh --single \
  --ref-repo https://github.com/example/repo-a.git \
  --ref-repo https://github.com/example/repo-b.git

# 云端评测需：
export SWE_BENCH_API_KEY=<key>
bash scripts/test_pipeline_e2e.sh --single

# 查看轨迹
python3 scripts/inspect_pipeline_traj.py trajectories/root/<run_dir>/
```

---

## 5. 已遇到问题与处理

### 5.1 早期问题（已修复）

| 现象 | 原因 | 处理 |
|------|------|------|
| `sb-cli` 401 / Missing x-api-key | 未配置 SWE-Bench 云端 API Key | 设置 `SWE_BENCH_API_KEY`，或脚本中无 Key 时 `evaluate=False` |
| `preds.json` 中 patch 为空 | 多因 **litellm 无法计价** 导致首步即 `ModelConfigurationError`，0 次真实 API 调用 | 所有模型 `per_instance_cost_limit` / `total_cost_limit` 置 **0** |
| 空 patch / 解析失败 | 部分模型对 function calling 支持差 | YAML 改为 `thought_action`；Planning 去掉 submit 审查 bundle |
| `inspect_pipeline_traj.py` 语法错误 | f-string 内错误转义引号 | 已改为普通变量拼接打印 |
| 验证轮次 `model_stats` 虚高 | 每轮把验证模型 **累计** stats 再加一遍 | 已改为 **增量**（snapshot + 差分） |
| 外部参考仓库同名覆盖 | 仅用 basename 命名目录，可能冲突 | 目录名改为 `basename + sha1短哈希` |
| 本地参考仓库链接偶发失效 | 目标路径已存在普通目录时 `ln -sfn` 不稳定 | 改为先 `rm -rf` 目标，再 `ln -s` |

### 5.2 第二轮 Review 发现的问题（已修复）

| 严重度 | 问题 | 原因 | 修复 |
|--------|------|------|------|
| **高** | `run-batch` 模式下 `PipelineAgent` 崩溃 | `run_batch.py:346` 执行 `agent.replay_config = single_run_replay_config`，但 `PipelineAgent` 缺少 `replay_config` 属性 | 在 `__init__` 中添加 `self.replay_config = None` |
| **高** | Planning 阶段异常导致整个实例失败 | `_run_planning_phase` 中模型返回格式错误或网络异常会直接向上抛出，终止整个实例 | 在 `run()` 中 catch Planning 异常，降级为"无 plan"继续 Coding |
| **高** | `_parse_score` 对 "X/10" 格式解析错误 | 正则匹配 "3/10" 产生 `["3", "10"]`，取 `numbers[-1]` 得到 10 而非 3，**低分被误判为满分**，导致错误地接受不合格 patch | 新增 `X/10` 专用正则优先匹配；对独立数字匹配用 word boundary 避免拆分 |
| **中** | `on_run_start()` hook 调用位置错误 | 原来在 `_run_planning_phase` 内部调用，若 Planning 阶段被异常跳过则 hook 永远不会触发 | 移至 `run()` 方法顶部，确保始终触发一次 |
| **低** | 验证消息缺少 `message_type` 字段 | 传给 `LiteLLMModel.query()` 的消息是简单 dict，不含 `HistoryItem` 必填的 `message_type`；运行时不报错但类型不一致 | 消息中补充 `message_type: "thought"` |

### 5.3 第三轮修复的设计问题

| 问题 | 原因 | 修复 |
|------|------|------|
| 验证 prompt 不含 `accept_score` | 模型不知道"几分算通过"，评分标准飘忽 | `VerificationConfig.instance_template` 默认模板和 YAML 均注入 `{{accept_score}}`，渲染时传入配置值 |
| `on_run_done` hook 传递最后一次 coding agent 数据 | 如果 best_submission 来自更早的 attempt，hooks 收到的数据与最终提交不一致 | 新增 `best_coding_agent` 变量追踪最佳 agent 实例，`on_run_done` 优先使用它 |
| Coding 重试时小模型看不到前一轮 patch | 验证反馈不够具体时小模型可能重复犯同一错误 | `_run_coding_phase` 新增 `previous_patch` 参数，在 `<plan>` 中注入前一轮 diff（标注 "YOUR PREVIOUS PATCH"） |
| `_get_best_coding_data` 与 `best_submission` 不对齐 | trajectory 按验证分数选，但 best_submission 按"分数高+非空"选，极端情况下来自不同 attempt | 新增 `_best_coding_attempt_idx`，在 `run()` 中更新 best_submission 时同步记录索引，`_get_best_coding_data` 直接使用该索引 |

---

## 6. 已知限制与设计注意

### 6.1 架构层面

1. **`PipelineAgent` 非 `AbstractAgent` 子类**：无 `step()` 方法；仅通过 `run()` 接入 `run-batch`。若未来有代码假设统一 Agent 接口（如统一监控、中断控制），需适配。
2. **每次 Coding 重试都重新安装 tools**：`DefaultAgent.setup()` 调用 `tools.install(env)` 会重新上传 bundle 和执行安装命令。在 3-4 次重试场景下会有可观的额外耗时（每次约数秒到十几秒）。`RetryAgent` 也有同样问题（它通过 `hard_reset` + 重新 setup），所以这是 SWE-agent 的通用限制。

### 6.2 验证阶段

3. **验证模型只看 diff 不看运行结果**：当前验证是纯静态的文本 review，无法捕捉运行时 bug（如测试不通过、import 错误）。要做运行验证需扩展为 agent 级别（让验证模型在容器内执行测试）。
4. **`n_samples > 1` 时取平均分但只用第一条 feedback**：若多次采样分数差异大，平均分可能不代表主流意见，且 feedback 仅取 `responses[0]`。

### 6.3 Pipeline 流程

5. **计划质量完全依赖模型**：若 Planning agent 未写 `plan.md` 且 thought 兜底也无有效内容，Coding agent 将在无指导下工作。已添加 warning 日志，但无更强的降级策略。
6. **软重置局限**：依赖 `repo.get_reset_commands()` 的 `git restore . && git reset --hard && git checkout <commit> && git clean -fdq`。若 agent 在仓库外写了文件（如 `/tmp/` 下的临时脚本），这些不会被清理，可能影响后续 attempt 的行为。

### 6.4 外部参考仓库

11. **参考仓库拉取成本**：按实例执行 startup 命令，batch 大规模运行会重复 clone/fetch，启动成本较高。
12. **本地路径语义**：`reference_repos` 里的绝对路径是"容器内路径"，不是宿主机路径。需要宿主机文件时应通过 Docker volume mount 或预先放入镜像。

---

## 7. 后续改进建议

### 短期（可立即实施）

- [x] 在 `VerificationConfig.instance_template` 中注入 `accept_score`，让验证模型知道通过标准。（已完成）
- [x] `on_run_done` 传递 best attempt 的 trajectory/info，而非 last attempt。（已完成）
- [x] Coding 重试时注入前一轮的 diff，让小模型知道之前做了什么。（已完成）
- [x] `_get_best_coding_data` 与 `best_submission` 对齐，使用统一的 `_best_coding_attempt_idx`。（已完成）
- [ ] 添加 `--dry-run` 模式到测试脚本，只打印将要执行的命令。

### 中期

- [ ] 继承或对齐 `AbstractAgent`，统一 `step()` / `logger` / 轨迹工具链。
- [ ] 为验证阶段增加可选的"运行测试"能力（让验证模型在容器内执行 `pytest` 等）。
- [ ] 支持 Planning 和 Coding 使用不同的 tool bundles 并优化重复安装。

### 长期

- [ ] 支持更灵活的多阶段配置（如 Planning → Code Review → Coding → Testing → Verification）。
- [ ] 支持 Planning 阶段输出结构化 plan（JSON/YAML），便于 Coding 阶段精确执行。
- [ ] 支持异步/并行验证（多个 coding attempt 并行生成，验证模型批量评审）。

---

## 8. 文件索引（快速查找）

```
sweagent/agent/agents.py              # PipelineAgentConfig, VerificationConfig, get_agent_from_config
sweagent/agent/extra/pipeline_agent.py # PipelineAgent 实现
sweagent/run/batch_instances.py       # SWEBenchInstances 扩展（reference_repos）
config/pipeline_qwen.yaml             # SGLang + Qwen3.5 示例配置
scripts/test_pipeline_e2e.sh          # 端到端测试脚本
scripts/inspect_pipeline_traj.py      # 轨迹检查工具
```

---

*最后更新：2026-03-19 — 第三轮修复，完成 4 项短期改进（accept_score 注入、on_run_done 对齐、前轮 diff 注入、trajectory/submission 对齐）。*
*文档随实现演进可继续增补；若行为与代码不一致，以仓库内源码为准。*
