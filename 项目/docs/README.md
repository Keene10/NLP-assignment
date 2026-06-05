# 实验流程说明

本项目按照原始功能目录保留代码，不再使用额外的临时实验目录。

## 数据处理

```bash
python -m rag.document.prepare_documents --input-dir 财报数据库/test --output-dir outputs/extracted_text_ocr --recreate --progress
```

作用：

- 读取 `财报数据库/test` 中的 PDF。
- 提取正文、表格和 OCR 补充文本。
- 按页导出文本，并切分为 chunks。

## 多模态图表说明补充

```bash
python -m rag.document.augment_chunks_with_chart_descriptions --chunks outputs/extracted_text_ocr/chunks.jsonl --chart-description-dir 归档/chart_descriptions --output-dir outputs/extracted_text_ocr_multimodal
```

作用：

- 读取多模态模型生成的图表/页面说明。
- 按文件名和页码追加到原始 chunks。
- 输出增强版 `outputs/extracted_text_ocr_multimodal/chunks.jsonl`。

## 构建向量库

```bash
python -m rag.vector.build_vector_db --chunks outputs/extracted_text_ocr_multimodal/chunks.jsonl --vector-db-path outputs/vector_db_multimodal --backend simple --recreate
```

作用：

- 使用 `m3e-small` 生成 embedding。
- 构建增强版向量库。

## 问答生成

```bash
python app/cli.py
```

作用：

- 读取 `财报数据库/test.json`。
- 使用本地 hybrid retrieval 和 reranker 选择 page。
- 调用 API 生成 answer。
- 将结果填回 `财报数据库/test.json`。

## 本地评测

如果存在 `财报数据库/test_ground_truth.json`，运行主流程后会自动输出：

```text
outputs/final_evaluation.json
```
