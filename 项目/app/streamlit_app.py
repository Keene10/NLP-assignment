import streamlit as st
import os
from rag.document.processor import DocumentProcessor
from rag.vector.vector_db import VectorDB
from rag.retriever.rag import RAGService

# 设置页面配置
st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🤖",
    layout="wide"
)

# 初始化服务
rag_service = RAGService()

# 侧边栏
with st.sidebar:
    st.title("文档管理")
    
    # 上传文档
    uploaded_files = st.file_uploader(
        "上传文档",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "md", "html"]
    )
    
    if uploaded_files:
        if st.button("处理并添加到知识库"):
            with st.spinner("处理文档中..."):
                processor = DocumentProcessor()
                vector_db = VectorDB()
                
                for file in uploaded_files:
                    # 保存临时文件
                    file_path = f"temp_{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # 处理文档
                    chunks = processor.process(file_path)
                    
                    # 添加到向量数据库
                    vector_db.add_documents(chunks)
                    
                    # 删除临时文件
                    os.remove(file_path)
                
                st.success("文档处理完成并添加到知识库！")

# 主界面
st.title("RAG智能问答系统")

# 聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = rag_service.answer(prompt)
            
            if isinstance(result, dict):
                st.markdown(result["answer"])
                if result["sources"]:
                    st.markdown("\n**参考来源：**")
                    for source in result["sources"]:
                        st.markdown(f"- {source}")
            else:
                st.markdown(result)
    
    # 添加助手消息
    st.session_state.messages.append({"role": "assistant", "content": result["answer"] if isinstance(result, dict) else result})
