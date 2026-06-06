# 论文修订说明

**原始论文标题：** TD3-Based Adaptive Navigation Control of UAVs for High-precision Expressway Service Area Mapping  
**修订论文标题：** TD3-Based Adaptive Navigation Control of UAVs for High-precision Aerial Mapping in Complex Terrain Environments  
**修订日期：** 2026-06-06  

---

## 一、总体修订策略

本次修订的核心目标是**提升论文的普适性与推广价值**，将研究背景从特定应用场景（高速公路服务区）扩展为更广泛的复杂地形无人机高精度测绘场景。具体策略如下：

1. **仅修改叙述性/框架性文本**：标题、摘要、关键词、引言中的背景与动机段落，不涉及任何技术内容。
2. **保留所有技术核心内容**：数学公式、实验数据与结果、图表、算法伪代码、参考文献、论文整体结构均保持不变。
3. **修改部分全部加粗显示**：在修订版 docx 文件中，所有改动文字均以**加粗**格式标注，便于审阅。

---

## 二、逐项修改对照表

| 位置 | 原文 | 修订后 | 修改说明 |
|------|------|--------|----------|
| **标题** | TD3-Based Adaptive Navigation Control of UAVs for High-precision **Expressway Service Area Mapping** | TD3-Based Adaptive Navigation Control of UAVs for High-precision **Aerial Mapping in Complex Terrain Environments** | 将特定场景"高速公路服务区"替换为通用场景"复杂地形环境" |
| **摘要 短语1**（Para 8） | high-precision map-ping tasks for mountainous **expressway service areas** | high-precision map-ping tasks for **complex mountainous terrain areas** | 去除高速公路服务区限定词，泛化为复杂山地地形 |
| **摘要 短语2**（Para 8） | applications in **complex expressway service-area environments** | applications in **complex terrain environments** | 去除高速公路服务区限定词，泛化为复杂地形环境 |
| **关键词**（Para 9） | **Expressway Service Area** | **Complex Terrain Aerial Mapping** | 以更通用的关键词替换特定场景关键词，提升检索覆盖范围 |
| **引言第1段**（Para 12） | 以自动驾驶与高速公路服务区为背景，列举加油站、便利店等设施 | 以自主系统（无人测绘平台、巡检机器人、物流无人机）对复杂地形高精度测绘的需求为背景 | 背景动机由特定场所替换为通用应用需求，覆盖基础设施巡检、精准农业、应急响应、物流等 |
| **引言第2段**（Para 13） | particularly for **expressways and their associated service areas** situated in mountainous terrains | particularly for **complex terrain areas** situated in mountainous regions | 去除高速公路限定，强调山地地形通用挑战 |
| **引言第5段**（Para 16） | challenging scenarios such as high-precision mapping of **expressway service areas in mountainous regions** | challenging scenarios such as **high-precision aerial mapping in complex mountainous terrain** | 泛化应用场景描述 |
| **引言第7段**（Para 18） | enhancing the suitability of UAV-based high-precision mapping for **expressway service areas** | enhancing the suitability of UAV-based **high-precision aerial mapping in complex terrain environments** | 结论句泛化，使适用性声明覆盖更广泛场景 |

---

## 三、针对审稿人"普适性不足"意见的逐点回应

### 审稿人意见（假设）
> "该研究的背景和应用场景仅限于高速公路服务区，普适性不足，难以推广至其他无人机导航控制场景。"

### 作者回应

**回应 1：技术方法本身具有普适性**  
本文提出的 TD3 自适应导航控制框架基于强化学习连续动作策略，其核心设计（实时调整航向与高度导航增益、锥形螺旋轨迹跟踪）并不依赖于高速公路服务区的任何特殊属性。该方法适用于所有需要在复杂地形中执行高精度低空测绘任务的固定翼无人机场景。

**回应 2：已修订论文定位以体现普适性**  
我们已将标题、摘要、关键词及引言中与高速公路服务区相关的特定场景描述，修订为"复杂地形环境中的高精度航空测绘"这一更广泛的应用定位。修订后的表述更准确地反映了本方法的适用范围，包括山地基础设施巡检、精准农业、应急响应及物流配送等领域。

**回应 3：实验结果的代表性**  
实验所用山地仿真环境（含高程变化、风干扰等条件）本身代表了复杂地形的典型特征，而非高速公路服务区的专属条件。仿真数据的有效性不因场景表述变化而受影响。

**回应 4：关键词优化提升检索覆盖**  
将关键词"Expressway Service Area"替换为"Complex Terrain Aerial Mapping"，有助于吸引更广泛研究领域（无人机测绘、地形感知导航、自主系统）的读者关注。

---

## 四、保留不变的内容清单

以下内容**完全未作修改**，保证论文技术严谨性与可复现性：

| 类别 | 具体内容 |
|------|----------|
| **数学公式** | UAV 动力学模型（Section 2.1）、3D-CSNFL 算法公式（Section 2.2）、TD3 算法更新公式（Section 3）等所有公式 |
| **实验数据与结果** | 所有 RMSE 数据、完成率统计、对比基线结果（FixedGain K=0.8 vs TD3）均保持原值 |
| **图表** | 所有实验图（轨迹图、误差曲线、收敛曲线等）及数据表格 |
| **算法伪代码** | TD3 训练流程伪代码 |
| **参考文献** | 全部文献引用格式与内容 |
| **论文结构** | 章节编号与标题（Section 2–5 及结论部分）均不变 |
| **作者信息** | 作者姓名、单位、通讯邮箱 |
| **仿真环境描述** | ArduPilot–Gazebo SITL 环境配置细节 |

---

## 五、投稿建议

1. **Cover Letter 中明确说明修订逻辑**：向编辑强调本次修订仅修改了应用场景的叙述定位，技术内容、实验结果及结论均未改变，确保审稿人对修订范围有清晰认知。

2. **在 Response Letter 中逐条回应**：对每一条审稿意见，对应本文修订表中的具体修改条目，提供"原文 → 修订后"的对照，便于审稿人快速核查。

3. **加粗标注辅助审阅**：修订版 docx 中所有改动文字均已加粗，无需另行提交 Track Changes 版本，但如期刊要求，可进一步生成 Track Changes 版本。

4. **关注后续扩展机会**：修订后的通用化定位为后续研究扩展（如将方法应用于林区巡检、电力线巡查、山地农业测绘等场景）奠定了更好的论文基础。
