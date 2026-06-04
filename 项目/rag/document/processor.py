from .loader import DocumentLoader
from .splitter import DocumentSplitter

class DocumentProcessor:
    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
        clean_text=True,
        extract_tables=True,
        table_mode='text',
        enable_ocr=False,
        ocr_cache_dir=None,
        ocr_dpi=144,
        ocr_min_confidence=0.5,
        ocr_progress=False,
    ):
        """
        初始化文档处理器
        :param chunk_size: 文本块大小
        :param chunk_overlap: 文本块重叠部分大小
        """
        self.loader = DocumentLoader(
            clean_text=clean_text,
            extract_tables=extract_tables,
            table_mode=table_mode,
            enable_ocr=enable_ocr,
            ocr_cache_dir=ocr_cache_dir,
            ocr_dpi=ocr_dpi,
            ocr_min_confidence=ocr_min_confidence,
            ocr_progress=ocr_progress,
        )
        self.splitter = DocumentSplitter(chunk_size, chunk_overlap)

    def load(self, file_path):
        """加载单个文档，不进行分割。"""
        return self.loader.load(file_path)

    def load_directory(self, directory_path, recursive=True, extensions=None):
        """批量加载目录文档，不进行分割。"""
        return self.loader.load_directory(directory_path, recursive, extensions)

    def split_documents(self, documents):
        """分割已经加载好的 LangChain Document 列表。"""
        return self.splitter.split(documents)

    def process(self, file_path):
        """处理文档：加载并分割"""
        # 加载文档
        documents = self.loader.load(file_path)
        # 分割文档
        chunks = self.splitter.split(documents)
        return chunks

    def process_directory(self, directory_path, recursive=True, extensions=None):
        """批量处理目录：加载并分割所有支持的文档。"""
        documents = self.load_directory(directory_path, recursive, extensions)
        return self.split_documents(documents)
