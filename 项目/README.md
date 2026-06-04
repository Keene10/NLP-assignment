# RAG 智能问答系统（含多模态图表增强）

本项目用于金融研报问答：读取 PDF 研报，完成 OCR/表格增强文本抽取、**多模态图表描述**、向量检索、候选页选择，并调用大模型生成答案。

## 目录结构

- `app/cli.py`：最终一键运行入口，会把结果直接填写到 `财报数据库/test.json`。
- `app/streamlit_app.py`：原始 Streamlit Web 界面。
- `rag/document/`：文档加载、PDF/OCR/表格抽取、文本切分。
- `rag/vector/`：嵌入模型和向量库封装。
- `rag/retriever/`：RAG 检索、证据页选择、Prompt 构建和回答生成。
- `config/config.py`：模型、向量库、路径等配置。
- `tests/`：基础测试。
- `财报数据库/`：测试问题、PDF 数据、标准答案。
- `outputs/`：中间产物（图表描述 JSON、页码计划、评测结果等）。
- **`scripts/`：新增工具脚本**
  - `detect_chart_pages.py`：检测 PDF 中包含图表/表格的页码。
  - `describe_charts_with_qwenvl.py`：调用 Qwen3-VL API 对图表页生成结构化描述。

## 环境准备

```bash
pip install -r requirements.txt
```

在 `.env` 中填写 API key（用于主流程的大模型调用）。

> **注意**：多模态图表描述需要阿里云百炼 API Key，请配置环境变量：
> ```bash
> export DASHSCOPE_API_KEY="sk-xxxx"
> ```

## 运行

### 方式一：主流程（已有文本增强）

```bash
python app/cli.py
```

程序会按顺序执行：

1. 读取 `财报数据库/test.json` 中的问题。
2. 使用本地 OCR 增强文本和向量库获得基础候选页。
3. 调用同一个 API 模型做"证据页选择"，只选页码，不生成答案。
4. 在选中页及相邻页上下文中生成答案。
5. 将 `filename`、`page`、`answer` 直接填回 `财报数据库/test.json`，保留原有 `question` 和 JSON 数组结构。
6. 如果存在 `财报数据库/test_ground_truth.json`，会在本地生成最新评测 `outputs/final_evaluation.json`。

常用参数：

```bash
python app/cli.py --resume
python app/cli.py --skip-page-selector
python app/cli.py --use-existing-selected-page-plan
```

### 方式二：多模态图表描述增强（新增）

针对 PDF 中的图表/表格页面，使用 Qwen3-VL 生成结构化描述，供 RAG 系统使用。

**Step 1：检测图表页**

```bash
python scripts/detect_chart_pages.py
```

- 解析 PDF 图表目录或扫描页面内联标记，识别包含图表的页码。
- 输出 `outputs/chart_pages.json`，记录每个 PDF 的图表页列表。

**Step 2：生成图表描述**

```bash
python scripts/describe_charts_with_qwenvl.py --model qwen3-vl-plus --dpi 150
```

- 将图表页渲染为图片，调用阿里云百炼 Qwen3-VL API。
- 生成结构化 JSON 描述（类型、标题、关键数据、趋势结论）。
- 结果保存在 `outputs/chart_descriptions/`。

> 首次运行前建议加 `--dry-run` 只生成截图，确认质量后再正式调用 API：
> ```bash
> python scripts/describe_charts_with_qwenvl.py --dry-run
> ```

## 多模态描述字段说明

每个图表的描述包含以下字段：

| 字段 | 说明 |
|------|------|
| `chart_type` | 图表类型：`table` / `bar_chart` / `line_chart` / `pie_chart` / `diagram` / `image` |
| `title` | 图表标题 |
| `one_liner` | 最精简的一句话总结（含关键数字），适合向量检索 |
| `key_entities` | 关键实体列表（数字、公司名、年份等） |
| `detailed_facts` | 较详细的事实描述，供生成答案时参考 |
| `trend_conclusion` | 数据趋势或结论 |

## 注意事项

1. **GitHub 不托管大文件**：`.gitignore` 已排除模型文件（`m3e-small/`）、PDF 截图（`chart_images/`）和 OCR 缓存。队友 clone 后需自行下载模型或重新运行脚本生成。
2. **免费额度**：Qwen3-VL-Plus 有百万级 Token 免费额度，230 页图表描述约消耗 60~70 万 Token，完全够用。
3. **页码检测策略**：不同 PDF 采用不同检测策略（目录解析 / 资料来源 / 内联标记），详见 `scripts/detect_chart_pages.py` 注释。
