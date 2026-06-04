from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        初始化文本分割器
        :param chunk_size: 文本块大小
        :param chunk_overlap: 文本块重叠部分大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n【表格抽取】",
                "\n\n【OCR补充】",
                "\n\n",
                "\n",
                "。",
                "；",
                "，",
                " ",
                "",
            ],
        )
    
    def split(self, documents):
        """分割文档"""
        return self.splitter.split_documents(documents)
    
    def split_text(self, text):
        """分割文本"""
        return self.splitter.split_text(text)
