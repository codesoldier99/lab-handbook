# 课题组周度推进 SOP（2026 v1.0）

## 0. 目标
- 高水平科研成果（论文/专利/竞赛/项目里程碑）
- 高质量就业/深造（作品集：代码+数据版本+实验记录+指标对比）

## 1. 核心机制（3件事）
1) GitHub = 唯一真相源（Issue/PR/Commit/实验单/周报都在这里）
2) 每周必须交：MVP（最小可验证成果）+ 周报（≤1页）
3) 每周三节点：周一计划、周三中检、周五冻结交付

## 2. MVP 一定要挂到“里程碑”
- 里程碑（学期/年度） → 月里程碑 → 周 MVP
- 每个周 MVP 在 Issue 中必须填写：归属 Milestone（例如 M1-数据闭环 / M2-MAP稳定 / M3-指标提升,M代表月份）
- 每人每月允许 1 个探索性 MVP（允许失败，但必须写复盘实验单）

## 3. 定义完成标准（DoD）
一个 MVP 完成，必须至少具备：
-结果证据（图/表/日志/指标）
- 可复现入口（scripts/run_demo.*）
- 实验单（experiments/*.md，含commit id、数据版本、参数）
-数据版本引用（data_manifest/*.md）

## 4. 周期节奏（硬性）
- 周一 12:00 前：开 1~3 个 Issue（本周目标）
- 周三 18:00 前：Issue 留言中检（卡点必须暴露）
- 周五 21:00 前：周报 + 结果 + 代码/脚本 + 实验单

## 5. 组会汇报（3~6分钟）
必须按顺序讲：
1) 一句话结论
2) 证据（图/表/指标）
3) 和谁比（基线/上周/文献）
4) 下周最关键一步

## 6. 表格模板
- 周报模板：docs/weekly/_template.md
- 实验单模板：experiments/_template.md
- 数据清单模板：data_manifest/_template.md
- Issue 模板：.github/ISSUE_TEMPLATE/mvp.yml

## 7. 一键运行（必须做 Level 0，鼓励 Level 1）
- Level 0：本地一键跑 scripts/run_demo.*
- Level 1：GitHub Actions 一键跑（C# / HALCON-self-hosted）
- Level 2：Release 可执行 Demo（可选）

