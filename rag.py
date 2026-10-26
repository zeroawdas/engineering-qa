"""RAG 核心模块：文档加载 → 切分 → 向量化 → 检索 → 生成。

每个函数对应 RAG 流水线的一步，命名和步骤一一对应，方便理解与讲解。
"""
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import dashscope
from dashscope import TextEmbedding

# ── 从 .env 读取配置 ──────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

QIANWEN_API_KEY = os.getenv("QIANWEN_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")


class DashScopeEmbeddings(Embeddings):
    """用阿里云 DashScope 的 text-embedding-v2 把文本转成向量。

    为什么用独立的 Embedding 服务？因为 DeepSeek 官方不提供 Embedding 接口，
    所以向量化交给 DashScope，生成回答交给 DeepSeek，两者各司其职。
    """

    def __init__(self):
        if not QIANWEN_API_KEY:
            raise ValueError("未设置 QIANWEN_API_KEY，请检查 .env 文件")
        dashscope.api_key = QIANWEN_API_KEY

    def embed_documents(self, texts):
        resp = TextEmbedding.call(model=EMBEDDING_MODEL, input=texts)
        if resp.status_code == 200:
            return [item["embedding"] for item in resp.output["embeddings"]]
        raise RuntimeError(f"Embedding 调用失败: {resp.message}")

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def load_documents(data_dir: str):
    """加载 data/ 目录下的 .txt / .md / .pdf 文档。"""
    docs = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        try:
            if ext in (".txt", ".md"):
                docs.extend(TextLoader(path, encoding="utf-8").load())
            elif ext == ".pdf":
                docs.extend(PyPDFLoader(path).load())
        except Exception as e:
            print(f"跳过 {fname}: {e}")
    return docs


def split_documents(docs, chunk_size=500, chunk_overlap=50):
    """按固定大小切分文本，重叠 50 字防止语义在边界处被截断。

    separators 里放中文标点，让切分尽量落在句子或段落的边界上，
    而不是硬生生在句子中间切断。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks, embeddings):
    """把切分好的文本块向量化，写入 FAISS 索引。"""
    return FAISS.from_documents(chunks, embeddings)


def retrieve(query, vector_store, top_k=3):
    """用向量相似度检索与问题最相关的 top_k 个文本块。"""
    return vector_store.similarity_search(query, k=top_k)


def build_llm():
    """创建 DeepSeek 对话模型。"""
    if not DEEPSEEK_API_KEY:
        raise ValueError("未设置 DEEPSEEK_API_KEY，请检查 .env 文件")
    return ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        model=DEEPSEEK_MODEL,
        temperature=0.3,
    )


# Prompt 的关键约束：只依据资料回答，资料里没有就直说，避免模型"幻觉"编造答案。
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是工程领域的知识助手，只根据下面给出的参考资料回答问题。\n"
     "如果资料中没有相关内容，请明确回答「资料中未找到相关内容」，不要编造。\n\n"
     "参考资料：\n{context}"),
    ("human", "{question}"),
])


def generate_answer(query, context_docs, llm):
    """把检索到的资料拼进 Prompt，让 LLM 基于资料生成答案。"""
    context = "\n\n---\n\n".join(
        f"[资料{i + 1}] {doc.page_content}" for i, doc in enumerate(context_docs)
    )
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})
