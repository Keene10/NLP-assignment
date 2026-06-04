# RAG 智能问答系统

本项目用于金融研报问答：读取 PDF 研报，完成 OCR/表格增强文本抽取、向量检索、候选页选择，并调用大模型生成答案。

## 目录结构

- `app/cli.py`：最终一键运行入口，会把结果直接填写到 `财报数据库/test.json`。
- `app/streamlit_app.py`：原始 Streamlit Web 界面。
- `rag/document/`：文档加载、PDF/OCR/表格抽取、文本切分。
- `rag/vector/`：嵌入模型和向量库封装。
- `rag/retriever/`：RAG 检索、证据页选择、Prompt 构建和回答生成。
- `config/config.py`：模型、向量库、路径等配置。
- `tests/`：基础测试。
- `财报数据库/`：测试问题和 PDF 数据。
- `outputs/`：必要中间产物，仅保留 OCR 文本、向量库、页码计划、最新 debug/评测。

## 运行

先在 `.env` 中填写 API key，然后运行：

```bash
python app/cli.py
```

程序会按顺序执行：

1. 读取 `财报数据库/test.json` 中的问题。
2. 使用本地 OCR 增强文本和向量库获得基础候选页。
3. 调用同一个 API 模型做“证据页选择”，只选页码，不生成答案。
4. 在选中页及相邻页上下文中生成答案。
5. 将 `filename`、`page`、`answer` 直接填回 `财报数据库/test.json`，保留原有 `question` 和 JSON 数组结构。
6. 如果存在 `财报数据库/test_ground_truth.json`，会在本地生成最新评测 `outputs/final_evaluation.json`。

常用参数：

```bash
python app/cli.py --resume
python app/cli.py --skip-page-selector
python app/cli.py --use-existing-selected-page-plan
```
