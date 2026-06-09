# RAG 智能问答系统

本项目用于金融研报问答：读取 PDF 研报，完成 OCR/表格/多模态图表说明增强、向量检索、BM25 检索、候选页重排序，并调用大模型生成最终答案。

## 目录结构

- `app/cli.py`：最终一键运行入口，会把 `filename/page/answer` 直接填回 `财报数据库/test_new.json`。
- `app/streamlit_app.py`：原始 Streamlit Web 页面。
- `config/config.py`：模型、路径、检索权重、reranker、Top-K 等配置。
- `rag/document/`：PDF 加载、OCR/表格提取、文本切分、多模态图表说明合并。
- `rag/vector/`：embedding 模型和向量库封装。
- `rag/retriever/`：RAG 主逻辑、本地 page 检索、reranker、prompt 构造、指标评测。
- `rag/llm/`：大模型 API 调用封装。
- `tests/`：基础测试。
- `归档/chart_descriptions/`：多模态模型生成的图表/页面说明，用于补充原始文本。
- `财报数据库/`：测试问题文件与本地数据。

## 当前正式流程

当前默认使用增强版数据：

- 文本数据：`outputs/extracted_text_ocr_multimodal/chunks.jsonl`
- 向量库：`outputs/vector_db_multimodal`

page 选择不调用 API，默认使用 `calibrated` 本地页码计划：

1. m3e embedding 向量检索。
2. BM25 关键词检索。
3. exact phrase、年份/数字/图表编号、业务词等规则加权。
4. 图表直达页、图表目录页惩罚、自动章节语义路由。
5. hybrid topK 候选页进入通用锚点校准，锚点包括年份、季度、图表编号、数字单位、财务指标、英文数字实体、引号短语、实体短语和列表形态。
6. 对细节型问题选择性降低前 1-3 页摘要/目录页权重，再做 topK anchor 校准和强锚点邻页校准。
7. `BAAI/bge-reranker-v2-m3` 只在泛分析题、默认题或低置信度题上辅助，并用 anchor guard 防止 reranker 覆盖证据更强的候选页。
8. 将 topK 页和多模态图表说明作为 answer 证据，调用 API 生成答案。

默认运行方式与本地实验保持一致：`app/cli.py` 会设置 `TRANSFORMERS_OFFLINE=1` 和
`HF_HUB_OFFLINE=1`，所以 `BAAI/bge-reranker-v2-m3` 不会联网下载，而是从本机
HuggingFace 缓存或项目内模型目录加载。若要把项目打包给队友完全复现，可以：

- 保持 `PAGE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3`，要求队友本机已有该模型缓存；
- 或把模型放到 `models/bge-reranker-v2-m3`，并将 `.env` 中的
  `PAGE_RERANKER_MODEL` 留空或改成 `./models/bge-reranker-v2-m3`。

`STRICT_LOCAL_RERANKER=1` 为默认值。reranker 缺失时程序会直接报错，避免静默降级后
跑出不可比的 page 结果。

## 运行方法

先根据 `.env.example` 创建 `.env`，填入自己的 API Key。

直接运行完整流程：

```bash
python app/cli.py
```

程序会读取 `财报数据库/test_new.json`，并把最终结果填回该文件，同时在本地生成 debug/evaluation 输出。

只做 page ablation 评测：
```bash
python app/page_ablation.py
```

该脚本会输出 A-F 六版 page 选择结果到 `outputs/page_ablation_test_new/`。

## 重新构建数据

如果需要从原始 PDF 重新构建：

```bash
python -m rag.document.prepare_documents --input-dir 财报数据库/test --output-dir outputs/extracted_text_ocr --recreate --progress
python -m rag.document.augment_chunks_with_chart_descriptions --chunks outputs/extracted_text_ocr/chunks.jsonl --chart-description-dir 归档/chart_descriptions --output-dir outputs/extracted_text_ocr_multimodal
python -m rag.vector.build_vector_db --chunks outputs/extracted_text_ocr_multimodal/chunks.jsonl --vector-db-path outputs/vector_db_multimodal --backend simple --recreate
```

## 评测

如果本地存在 `财报数据库/test/test_new_ground_truth.json`，运行 `app/cli.py` 后会自动输出 ROUGE-1、ROUGE-2、BLEU 等指标到 `outputs/final_evaluation.json`。

## GitHub 注意事项

不要上传 `.env`、API Key、`outputs/`、本地模型目录、PDF 原始数据或向量库。仓库只需要保留代码、配置模板、说明文档和必要的小型辅助数据。
