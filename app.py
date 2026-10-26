"""Streamlit 界面：上传/展示知识库文档，输入问题，展示答案与引用来源。"""
import os

import streamlit as st

from rag import (
    DashScopeEmbeddings,
    load_documents,
    split_documents,
    build_vector_store,
    retrieve,
    build_llm,
    generate_answer,
)

st.set_page_config(page_title="工程知识库问答", layout="wide")
st.title("工程规范知识库问答系统")

DATA_DIR = "data"

# ── 侧边栏：参数 + 文档列表 ──────────────────────────────
with st.sidebar:
    st.header("检索参数")
    chunk_size = st.slider("分块大小", 200, 1000, 500, 50)
    chunk_overlap = st.slider("重叠长度", 0, 200, 50, 10)
    top_k = st.slider("检索数量 top_k", 1, 10, 3)

    st.divider()
    st.header("知识库文档")
    if os.path.isdir(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR)
                 if f.lower().endswith((".txt", ".md", ".pdf"))]
        if files:
            for f in files:
                st.text(f"- {f}")
        else:
            st.warning("data/ 目录暂无文档")
    else:
        st.warning("data/ 目录不存在")

    if st.button("重新构建索引", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# ── 加载 → 切分 → 向量化（用 cache_resource 缓存，避免每次交互重建）──
@st.cache_resource
def get_vector_store(chunk_size, chunk_overlap):
    embeddings = DashScopeEmbeddings()
    docs = load_documents(DATA_DIR)
    if not docs:
        return None, 0
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    vector_store = build_vector_store(chunks, embeddings)
    return vector_store, len(chunks)


vector_store, n_chunks = get_vector_store(chunk_size, chunk_overlap)

if vector_store is None:
    st.error("data/ 目录没有找到文档，请先放入工程规范 .txt/.md/.pdf 文件后重启。")
    st.stop()

st.info(f"已加载知识库：共 {n_chunks} 个文本片段")

# ── 问答区域 ─────────────────────────────────────────────
question = st.chat_input("请输入关于工程规范的问题...")

if question:
    with st.spinner("正在检索知识库..."):
        context_docs = retrieve(question, vector_store, top_k)

    with st.spinner("正在生成回答..."):
        llm = build_llm()
        answer = generate_answer(question, context_docs, llm)

    st.markdown("### 回答")
    st.write(answer)

    st.markdown("### 参考来源")
    if not context_docs:
        st.write("未检索到相关内容")
    for i, doc in enumerate(context_docs, 1):
        source = doc.metadata.get("source", "未知")
        with st.expander(f"资料 {i} · {os.path.basename(source)}"):
            st.write(doc.page_content)
