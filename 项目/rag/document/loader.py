import json
from pathlib import Path
import re

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

class DocumentLoader:
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.md', '.html'}

    def __init__(
        self,
        clean_text=True,
        extract_tables=True,
        table_mode='text',
        enable_ocr=False,
        ocr_cache_dir=None,
        ocr_dpi=144,
        ocr_min_confidence=0.5,
        ocr_progress=False,
    ):
        self.clean_text = clean_text
        self.extract_tables = extract_tables
        self.table_mode = table_mode
        self.enable_ocr = enable_ocr
        self.ocr_cache_dir = Path(ocr_cache_dir) if ocr_cache_dir else None
        self.ocr_dpi = ocr_dpi
        self.ocr_min_confidence = ocr_min_confidence
        self.ocr_progress = ocr_progress
        self._ocr_engine = None

    def load(self, file_path):
        """加载文档，根据文件类型选择不同的加载器"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == '.pdf':
            return self._load_pdf(file_path)
        elif ext == '.txt':
            documents = [Document(page_content=self._read_text_file(file_path), metadata={})]
            return self._with_metadata(documents, file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(str(file_path))
        elif ext == '.md':
            loader = UnstructuredMarkdownLoader(str(file_path))
        elif ext == '.html':
            loader = UnstructuredHTMLLoader(str(file_path))
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        return self._with_metadata(loader.load(), file_path)

    def _load_pdf(self, file_path):
        loader = PyPDFLoader(str(file_path))
        documents = self._with_metadata(loader.load(), file_path)
        table_pages = self._extract_tables(file_path, documents) if self.extract_tables else {}
        ocr_pages = self._extract_pdf_ocr(file_path, documents) if self.enable_ocr else {}
        image_counts = self._count_pdf_images(file_path, len(documents))

        for document in documents:
            metadata = document.metadata or {}
            page_number = metadata.get('page_number') or metadata.get('page')
            base_text = document.page_content or ''
            table_texts = table_pages.get(page_number, [])
            ocr_texts = ocr_pages.get(page_number, [])
            enhanced_sections = [self._clean_document_text(base_text)]

            if table_texts:
                enhanced_sections.append(self._format_section('表格抽取', table_texts))
            if ocr_texts:
                enhanced_sections.append(self._format_section('OCR补充', ocr_texts))

            document.page_content = '\n\n'.join(section for section in enhanced_sections if section.strip())
            metadata['content_type'] = 'enhanced_page'
            metadata['table_count'] = len(table_texts)
            metadata['has_tables'] = bool(table_texts)
            metadata['table_mode'] = self.table_mode if self.extract_tables else 'none'
            metadata['ocr_line_count'] = sum(len(text.splitlines()) for text in ocr_texts)
            metadata['has_ocr'] = bool(ocr_texts)
            metadata['image_count'] = image_counts.get(page_number, 0)
            metadata['has_images'] = metadata['image_count'] > 0
            metadata['content_sources'] = [
                source for source, enabled in (
                    ('text', bool(base_text.strip())),
                    ('table', bool(table_texts)),
                    ('ocr', bool(ocr_texts)),
                )
                if enabled
            ]
            document.metadata = metadata

        return documents

    def _read_text_file(self, file_path):
        for encoding in ('utf-8', 'utf-8-sig', 'gbk', 'gb18030'):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return file_path.read_text(encoding='utf-8', errors='ignore')

    def _extract_tables(self, file_path, documents):
        if self.table_mode == 'pdfplumber':
            return self._extract_pdfplumber_tables(file_path, len(documents))
        return self._extract_text_table_blocks(documents)

    def _extract_pdfplumber_tables(self, file_path, page_count):
        try:
            import pdfplumber
        except ImportError:
            return {}

        table_pages = {page_number: [] for page_number in range(1, page_count + 1)}
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables() or []
                    for table_index, table in enumerate(tables, start=1):
                        markdown = self._table_to_markdown(table)
                        if markdown:
                            table_pages[page_index].append(f'【表格 {table_index}】\n{markdown}')
        except Exception as exc:
            table_pages.setdefault(1, []).append(f'【表格抽取失败】{exc}')

        return table_pages

    def _extract_text_table_blocks(self, documents):
        table_pages = {}
        for document in documents:
            metadata = document.metadata or {}
            page_number = metadata.get('page_number') or metadata.get('page')
            blocks = self._find_table_like_blocks(document.page_content)
            if blocks:
                table_pages[page_number] = [
                    f'【疑似表格 {index}】\n{block}'
                    for index, block in enumerate(blocks, start=1)
                ]
        return table_pages

    def _find_table_like_blocks(self, text):
        lines = [self._clean_document_line(line) for line in (text or '').splitlines()]
        blocks = []
        current = []

        for line in lines:
            if self._looks_like_table_line(line):
                current.append(line)
            else:
                if len(current) >= 2:
                    blocks.append('\n'.join(current[:40]))
                current = []

        if len(current) >= 2:
            blocks.append('\n'.join(current[:40]))

        return blocks[:6]

    def _looks_like_table_line(self, line):
        if not line or len(line) > 220:
            return False
        numbers = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?', line)
        if len(numbers) >= 3:
            return True
        financial_keywords = (
            '收入', '利润', '净利', '毛利', '费用', '成本', '资产', '负债',
            '现金', '增长率', 'ROE', 'EPS', 'PE', 'PB', '亿元', '百万元',
        )
        return len(numbers) >= 2 and any(keyword in line for keyword in financial_keywords)

    def _table_to_markdown(self, table):
        rows = []
        max_columns = 0
        for row in table or []:
            cleaned_row = [self._clean_cell(cell) for cell in (row or [])]
            if any(cleaned_row):
                rows.append(cleaned_row)
                max_columns = max(max_columns, len(cleaned_row))

        if len(rows) < 2 or max_columns == 0:
            return ''

        normalized_rows = [row + [''] * (max_columns - len(row)) for row in rows]
        header = normalized_rows[0]
        body = normalized_rows[1:]
        separator = ['---'] * max_columns
        markdown_rows = [header, separator] + body
        return '\n'.join('| ' + ' | '.join(row) + ' |' for row in markdown_rows)

    def _clean_cell(self, cell):
        text = '' if cell is None else str(cell)
        return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

    def _extract_pdf_ocr(self, file_path, documents):
        try:
            import fitz
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            return {}

        if self._ocr_engine is None:
            self._ocr_engine = RapidOCR()

        ocr_pages = {}
        pdf = None
        try:
            pdf = fitz.open(str(file_path))
            zoom = self.ocr_dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            for document in documents:
                metadata = document.metadata or {}
                page_number = metadata.get('page_number') or metadata.get('page')
                cached_lines = self._load_cached_ocr(file_path, page_number)
                if cached_lines is not None:
                    if cached_lines:
                        ocr_pages[page_number] = ['\n'.join(cached_lines)]
                    if self.ocr_progress:
                        print(f'OCR cache hit: {file_path.name} page {page_number}', flush=True)
                    continue

                page_index = int(page_number) - 1
                if self.ocr_progress:
                    print(f'OCR start: {file_path.name} page {page_number}', flush=True)
                page = pdf.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_bytes = pixmap.tobytes('png')
                result, _ = self._ocr_engine(image_bytes)
                lines = self._filter_ocr_lines(result, document.page_content)
                self._save_cached_ocr(file_path, page_number, lines)
                if self.ocr_progress:
                    print(f'OCR done: {file_path.name} page {page_number}, lines={len(lines)}', flush=True)
                if lines:
                    ocr_pages[page_number] = ['\n'.join(lines)]
        finally:
            if pdf is not None:
                pdf.close()

        return ocr_pages

    def _ocr_cache_path(self, file_path, page_number):
        if self.ocr_cache_dir is None:
            return None
        file_dir = self.ocr_cache_dir / self._safe_name(file_path.stem)
        return file_dir / f'page_{int(page_number):03d}.json'

    def _load_cached_ocr(self, file_path, page_number):
        cache_path = self._ocr_cache_path(file_path, page_number)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return None
        return data.get('lines', [])

    def _save_cached_ocr(self, file_path, page_number, lines):
        cache_path = self._ocr_cache_path(file_path, page_number)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'filename': file_path.name,
            'page_number': int(page_number),
            'line_count': len(lines),
            'lines': lines,
        }
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    def _filter_ocr_lines(self, result, existing_text):
        if not result:
            return []

        existing_norm = self._normalize_for_dedupe(existing_text)
        seen = set()
        lines = []
        for item in result:
            if len(item) < 3:
                continue
            text = self._clean_document_line(item[1])
            confidence = item[2]
            if not text or confidence < self.ocr_min_confidence:
                continue
            text_norm = self._normalize_for_dedupe(text)
            if not text_norm or text_norm in seen or text_norm in existing_norm:
                continue
            seen.add(text_norm)
            lines.append(text)
        return lines

    def _count_pdf_images(self, file_path, page_count):
        try:
            import fitz
        except ImportError:
            return {}

        image_counts = {}
        pdf = None
        try:
            pdf = fitz.open(str(file_path))
            for page_index in range(min(page_count, pdf.page_count)):
                page = pdf.load_page(page_index)
                image_counts[page_index + 1] = len(page.get_images(full=True))
        finally:
            if pdf is not None:
                pdf.close()
        return image_counts

    def _format_section(self, title, items):
        body = '\n\n'.join(item.strip() for item in items if item.strip())
        if not body:
            return ''
        return f'【{title}】\n{body}'

    def _clean_document_text(self, text):
        if not self.clean_text:
            return text or ''

        lines = []
        for line in (text or '').splitlines():
            cleaned = self._clean_document_line(line)
            if cleaned and not self._is_noise_line(cleaned):
                lines.append(cleaned)

        return '\n'.join(self._dedupe_consecutive_lines(lines))

    def _clean_document_line(self, line):
        line = (line or '').replace('\u00a0', ' ').replace('\u3000', ' ')
        line = re.sub(r'\[Table_[^\]]+\]', ' ', line)
        line = line.replace('Y oY', 'YoY').replace('RO E', 'ROE')
        line = re.sub(r'[ \t]+', ' ', line)
        line = re.sub(r'\s+([，。；：！？、）】])', r'\1', line)
        line = re.sub(r'([（【])\s+', r'\1', line)
        return line.strip()

    def _is_noise_line(self, line):
        if line in {'[', ']', '(', ')'}:
            return True
        if len(line) <= 2 and re.fullmatch(r'[\d\s\-—_]+', line):
            return True
        if re.fullmatch(r'[A-Za-z0-9]{40,}', line):
            return True
        noise_patterns = [
            r'^请阅读.*免责声明',
            r'^敬请.*阅读.*免责声明',
            r'^证券研究报告$',
            r'^公司研究$',
            r'^\[Table_[^\]]+\]$',
        ]
        return any(re.search(pattern, line) for pattern in noise_patterns)

    def _dedupe_consecutive_lines(self, lines):
        result = []
        previous = None
        for line in lines:
            current = self._normalize_for_dedupe(line)
            if current and current == previous:
                continue
            result.append(line)
            previous = current
        return result

    def _normalize_for_dedupe(self, text):
        return re.sub(r'\W+', '', text or '').lower()

    def _safe_name(self, name):
        return ''.join(char if char not in '<>:"/\\|?*' else '_' for char in name)

    def load_directory(self, directory_path, recursive=True, extensions=None):
        """批量加载目录中的文档，默认递归加载支持的文件类型。"""
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        if not directory_path.is_dir():
            raise NotADirectoryError(f"不是目录: {directory_path}")

        allowed_extensions = {
            ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
            for ext in (extensions or self.SUPPORTED_EXTENSIONS)
        }

        pattern = '**/*' if recursive else '*'
        file_paths = sorted(
            path for path in directory_path.glob(pattern)
            if path.is_file() and path.suffix.lower() in allowed_extensions
        )

        documents = []
        for file_path in file_paths:
            documents.extend(self.load(file_path))
        return documents

    def _with_metadata(self, documents, file_path):
        """统一补齐文件名、页码等元数据，供检索和评测定位使用。"""
        ext = file_path.suffix.lower()
        total_pages = len(documents) if ext == '.pdf' else None

        for index, document in enumerate(documents, start=1):
            metadata = dict(document.metadata or {})
            original_page = metadata.get('page')
            page_label = metadata.get('page_label')

            metadata.update({
                'source': str(file_path),
                'filename': file_path.name,
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_type': ext.lstrip('.'),
            })

            if ext == '.pdf':
                metadata['page_index'] = original_page if isinstance(original_page, int) else index - 1
                metadata['page_number'] = index
                metadata['page'] = index
                metadata['total_pages'] = total_pages
                if page_label is not None:
                    metadata['page_label'] = str(page_label)
            elif self.clean_text:
                document.page_content = self._clean_document_text(document.page_content)

            document.metadata = metadata

        return documents
