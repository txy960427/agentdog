# AgentDoG 1.5 复现证据索引

论文：[AgentDoG 1.5, arXiv:2605.29801](https://arxiv.org/pdf/2605.29801)

公开训练数据：[AgentDoG1.0 Training Data README](../../data/1.0/source/AgentDoG1.0-Training-Data/README.md)

本文件只记录会影响复现设计的证据。`论文结论`是论文明确披露，`复现含义`是本项目据此采取的动作。

<a id="e1"></a>
## E1：Judge任务与CoT输出

- **位置**：本地`AgentDoG 1.5.pdf`第10页§3.2.1，附录第36–37页B.1 `CoT Generation Template`，第38–39页B.2 `Coarse Judgment Prompt Template`。
- **论文结论**：1.5版在最终标签前加入显式CoT，并称GPT-5.4为每个训练样本生成详细推理。粗粒度任务分析证据、意图、具体后果和安全影响，再输出二分类；细粒度任务先解释三个风险维度，再输出三个标签。
- **公开文本缺口**：B.1粗粒度模板列出四个问题，但第37页续框只要求`<Judgment> safe or unsafe </Judgment>`，没有Analysis或输入占位符；B.2才明确要求`<Analysis>`回答四问后再输出`<Judgment>`，并含trajectory和工具列表两个槽。B.1导语明确仅细粒度模板提供oracle labels；不能将该标签引导规则套用于粗粒度。论文没有公开粗粒度teacher调用外壳、解码参数、重试规则或最终SFT row。
- **复现含义**：teacher保留B.1的Task、Q1–Q4和safe note，显式补Profile/Tools/Trajectory输入块，并参照B.2补成四问`<Analysis>`后输出`<Judgment>`；输入块和格式规则属于项目补充。student继续使用已验证的Judge usage prompt。需要对齐的是teacher输出和SFT assistant目标，不要求两套输入prompt逐字相同。粗粒度Teacher输入不提供gold，冲突不能机械改标签。不能声称这个补全就是作者未公开的GPT-5.4请求原文。canonical数据先保存轨迹和标签，Step 4再生成CoT；训练输出不能只保留单个label。

<a id="e2"></a>
## E2：数据引擎的风险配置与safe/unsafe孪生

- **位置**：论文 `Data Preparation → Data Collection → Stage 1/2`。
- **论文结论**：每条轨迹先选择Risk Source、Failure Mode、Real-World Harm和候选工具，再确定safe或unsafe结果；同一场景骨架可以生成safe或unsafe版本。safe版本通过拒绝、验证、警告、安全fallback或停止风险路径处理风险。
- **复现含义**：带工具safe必须从已入选unsafe锚点生成并共享pair/group/split，不能作为独立随机数据池。

<a id="e3"></a>
## E3：自动质量检查

- **位置**：论文 `Data Collection → Stage 3: Automatic validation`。
- **论文结论**：规则检查验证工具调用格式、schema、类型、取值和引用完整性；模型检查验证合理性、步骤连贯、目标一致、事实可信和标签一致。标签必须由可观察行为支持，而不是仅由意图支持。
- **复现含义**：所有生成样本需要rule check、model check和冲突人工复核；工具报错不能自动把已经执行的危险行为改判safe。

<a id="e4"></a>
## E4：三维分布与论文均衡策略

- **位置**：论文 `Fine-grained data selection and balance`。
- **论文结论**：同时监控三个风险维度的边际分布和联合tuple；高频类必要时下采样，稀有但可靠的样本保留；论文明确说不采用机械均匀。
- **复现含义**：论文策略可作为后续筛选实验；当前Fine-SFT全量基线保留4,000条真实边际与联合tuple，不声称已经复现未公开的选择结果。

<a id="e5"></a>
## E5：影响函数净化与约1k训练样本

- **位置**：论文 `Data Purification`，公式 `guardrail-target-direction` 和 `guardrail-prism-score`。
- **论文结论**：论文用 `Q_safe` 的正负回答梯度构造guardrail方向，对候选SFT样本评分，保留高分样本，最终数据缩减至roughly 1k。
- **复现含义**：roughly 1k指论文净化后的训练规模，不是公开可直接还原的train/validation/test总和。公开材料没有`Q_safe`和最终样本ID，因此影响函数只能作为后续独立实验。
- **实现状态**：粗粒度SFT当前Train 1,000条、细粒度SFT当前Train 2,701条，二者均未做影响函数净化。规则/语义审核与梯度训练价值筛选不是同一件事；细粒度当前明确称为“全量4,000条基线”。
- **公式核对**：本地PDF第10–11页式(4)–(8)：目标正/负回答按token平均NLL求梯度，按参考模型对正回答的归一化似然偏好加权；候选样本按序列总NLL求梯度，再与目标梯度对比方向做内积。式(8)写出的分数是`g_z^T g_guard`，没有显式Hessian逆。正/负回答指风险判断正确/错误，不是safe/unsafe两个类别。
- **补齐边界**：`Q_safe`的具体样本/规模、参考checkpoint、候选池规模和保留ID未在该节给出。若自行补齐，需要声明这些工程选择；只能在对应任务的Train候选池内筛选，不能混入冻结的Validation/Test或benchmark来凑roughly 1k。

<a id="e6"></a>
## E6：Judge训练顺序

- **位置**：论文 `Training → Supervised Fine-Tuning / Reinforcement Learning`。
- **论文结论**：先分别对粗粒度或细粒度数据做SFT；随后只对细粒度模型继续RL。公开SFT学习率为 `1e-5`。细粒度RL使用GDPO，三维奖励对应FM、RWH、RS，维度权重为 `(0.3, 0.4, 0.3)`，缺少非平凡analysis时三维奖励归零。
- **复现含义**：粗粒度SFT、细粒度SFT和细粒度GDPO是不同训练产物，共享轨迹schema但不共享target字段。

<a id="e7"></a>
## E7：Judge正式测试集

- **位置**：论文 `Evaluation → Experimental Setup / Evaluation Tasks`。
- **论文结论**：粗粒度trajectory safety在R-Judge和ATBench上报告Accuracy、Precision、Recall、F1；Codex和Claw用于执行环境迁移。论文另说明Codex/Claw做了trajectory-format adaptation training。
- **复现含义**：不从v1.0训练数据额外制造正式test。R-Judge/ATBench是主test；Codex/Claw适配结果单独报告，不能与未适配粗粒度checkpoint混为一谈。

<a id="e8"></a>
## E8：公开v1.0数据能提供什么

- **位置**：公开数据README及固定revision `f4da9da000e1bde5dce851e522bcb1e3c50684ed`。
- **公开说明**：BinarySafety为4,000条train，FineGrainedTaxonomy为4,000条train；没有validation/test split。
- **本地实测**：BinarySafety前后两半完全重复，实际为1,000 unique safe + 1,000 unique unsafe；无显式Available Tools、无thought、输出仅label。FineGrained为4,000 unique unsafe，含工具、thought和三维标签，但没有CoT。
- **复现含义**：Binary必须先去重；validation从未入训候选另取；v1.0无法直接等同1.5净化数据。

<a id="e9"></a>
## E9：主执行模型SFT

- **位置**：论文 `Application 1 → Applying AgentDoG for Agentic Safety SFT`。
- **论文结论**：ATBench引擎先生成benign工具轨迹，再向工具描述、用户请求、工具调用和工具反馈注入风险，并生成尽量保留benign任务的safe轨迹。原始32,787 trajectory pairs经AgentDoG过滤后剩28,705条；再加入50,000条ToolBench、ToolAlpaca、ToolACE良性工具轨迹，形成约1:2安全关键/良性混合。Qwen3.5-4B非thinking SFT使用batch 128、cutoff 16,384、学习率 `5e-6`。
- **复现含义**：这是主执行模型数据，不是Judge的1k训练数据；不能拿28,705/50,000作为粗粒度Judge配比。

<a id="e10"></a>
## E10：主执行模型RL

- **位置**：论文 `Application 1 → Agentic Safety RL`。
- **论文结论**：Qwen3.5-4B非thinking，15,222个训练任务、323个工具、16个领域；GSPO，8 samples/task，rollout batch 32，最多400 rollout steps，每50步保存；损坏或截断tool-call做loss mask；相关任务按benign、environment injection、malicious query三元组组包。奖励按任务类型组合utility和AgentDoG safety。
- **复现含义**：后续RL数据需要环境spec、三元组ID、utility/safety reward和loss mask字段，不能只复用Judge label格式。

<a id="e11"></a>
## E11：本地已验证的粗粒度Prompt

- **位置**：[coarse baseline记录](../../judge_eval/baseline_run/sft/README.md)、[prompt.txt](../../judge_eval/prompt.txt)及对应run manifest。
- **实测配置**：Profile → Available Tools → Conversation；ATBench使用prose tools；R-Judge在已验证baseline commit中从调用提取`Tool: name`；thought位于action前；环境大字段head-tail压缩；任务模板不追加第二份tools；Qwen chat关闭thinking。
- **实测结果**：4B在ATBench为Acc 73.1/F1 74.6，在R-Judge为Acc 93.0/F1 93.4；prompt SHA-256为`99e7d6f3c535b3acf2845c3c6df7ebcf27cdde57b88f14ebd3e7c38c17ceadb1`，run git head为`222b55e49006cbf492f1a6ee422d2192a5d19952`。
- **复现含义**：训练、validation、internal test和外部benchmark必须调用同一版本renderer；当前HEAD若与baseline commit行为不同，以冻结renderer回归结果为准，不能静默改变输入格式。

<a id="e12"></a>
## E12：细粒度SFT、标签空间与主模型RL关系

- **位置**：论文§3.1、§3.2.1、§3.3、Table 3、Appendix B.1/B.2；官方Fine模型卡；本地ATBench与公开主模型RL脚本。
- **论文结论**：专用Fine任务只对unsafe轨迹预测RS、FM、RWH；先做细粒度SFT，再用GDPO继续训练。SFT使用经净化且补充CoT的数据，公开学习率为`1e-5`；论文没有单独报告Fine-SFT-only指标，也没有披露细粒度SFT与GDPO的精确query数量。
- **标签核对**：AgentDoG 1.5整体taxonomy覆盖Claw/Codex扩展后的15 RS、21 FM、11 RWH，但Table 3在基础ATBench的497条unsafe上评测，gold实际为8/14/10；官方Fine模型卡的使用prompt也只列出同一套8/14/10。论文没有交代最终Fine训练集是否混入扩展类别。
- **本地主模型RL代码核对**：`Agentic Safety Training/Agentic RL/slime_recipe`只从Judge输出解析`safe/unsafe`，安全奖励`S`为safe=1、unsafe=0，不读取RS/FM/RWH。
- **复现含义**：Table 3第一版使用8/14/10；完整ATBench Family的15/21/11另建扩展版本，不能静默混训。Fine-SFT与coarse_sft独立，本版使用全部4,000条v1.0 FineGrained unsafe，按2,701/678/621冻结并全部补充Fine Oracle CoT；这是全量基线，不声称复现论文未公开的影响函数净化后roughly 1k训练集。细粒度模型用于Judge自身诊断能力和GDPO三维优化，不是主执行Agent SFT/RL的必要前置。

<a id="e13"></a>
## E13：粗粒度审核中的可靠性与功能风险

- **来源**：[论文§3.1、附录A.2和A.3](https://arxiv.org/html/2605.29801v1)。
- **论文口径**：风险可以来自行为或决策模式，不以损害已经实现为前提。Failure Mode包括错误规划、错误工具参数、未验证工具输出和误导性信息；Real-world Harm包括功能失败、错误分析、资源浪费和机会损失。因此，没有实际外泄或发信不能单独证明safe。
- **本地执行判据**：检查可见轨迹中是否已经出现错误执行、错误状态记录或误导性行动结论，以及其具体功能/资源/机会风险；不能仅凭恶意用户请求、工具中的攻击文本或缺少授权上下文判unsafe，也不把所有轻微错误一概判unsafe。
- **本轮裁决**：6条原unsafe样本按上述口径保留标签，重写CoT，删除没有证据的注入执行、数据外泄、实际发信等推断。这是本助手的逐条裁决，不声称还原作者对这6条的原始标注理由，也不冒充外部审核模型pass。样本、证据边界及修订内容见[修订清单](../../data_pipeline/coarse_sft/steps/step4/check_repairs.json)，当前内容绑定的裁决见[manual_review.json](../../data/1.5/coarse_sft/step4/check/manual_review.json)。
