import os
import tempfile
import pytest
from rag.document.processor import DocumentProcessor

class TestDocumentProcessor:
    def test_process_text_file(self):
        """测试处理文本文件"""
        # 创建临时文本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("测试文档内容\n这是第二行")
            temp_file = f.name
        
        try:
            processor = DocumentProcessor()
            chunks = processor.process(temp_file)
            assert len(chunks) > 0
            assert "测试文档内容" in chunks[0].page_content
        finally:
            os.unlink(temp_file)
    
    def test_process_pdf_file(self):
        """测试处理PDF文件"""
        # 注意：需要安装pypdf库才能测试PDF文件
        pass