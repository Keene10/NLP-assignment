import os
import tempfile
import pytest
from rag.document.processor import DocumentProcessor
from rag.vector.vector_db import VectorDB
from rag.retriever.rag import RAGService

class TestRAGService:
    def setup_method(self):
        """设置测试环境"""
        # 创建临时文本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("苹果是一种水果，富含维生素C。")
            self.test_file = f.name
        
        # 处理文档并添加到向量数据库
        processor = DocumentProcessor()
        vector_db = VectorDB()
        chunks = processor.process(self.test_file)
        vector_db.create(chunks)
    
    def teardown_method(self):
        """清理测试环境"""
        # 删除临时文件
        os.unlink(self.test_file)
        # 删除向量数据库
        if os.path.exists('./vector_db'):
            import shutil
            shutil.rmtree('./vector_db')
    
    def test_answer(self):
        """测试RAG服务的回答功能"""
        rag_service = RAGService()
        result = rag_service.answer("苹果有什么营养？")
        
        # 验证结果格式
        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'sources' in result
        
        # 验证回答中包含相关信息
        assert "维生素C" in result['answer']