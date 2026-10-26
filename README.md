# 工程规范知识库问答系统（RAG）

一个基于 **RAG（检索增强生成）** 的工程规范知识库问答系统。

上传建筑、施工、验收等工程规范文档后，系统会自动完成「切分 → 向量化 → 建索引」，用户提问时先检索最相关的规范条文，再让大模型**基于条文**生成回答，从根本上减少大模型"凭空编造"的幻觉问题。

## 技术栈

| 层 | 选型 |
|----|------|
| 界面 | Streamlit |
| 编排框架 | LangChain |
| 向量数据库 | FAISS |
| Embedding | 阿里云 DashScope `text-embedding-v2` |
| 生成模型 | DeepSeek `deepseek-chat` |

## 功能

- 上传 `.txt` / `.md` / `.pdf` 工程规范文档
- 自动切分、向量化、建立 FAISS 索引
- 向量相似度检索 top_k 条最相关条文
- 基于检索结果生成回答，并展示引用来源

## 快速开始

**准备**：DeepSeek API Key、阿里云 DashScope API Key（用于 Embedding）

```bash
git clone <你的仓库地址>
cd engineering-qa

pip install -r requirements.txt

cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 QIANWEN_API_KEY

# 把工程规范文档放进 data/ 目录
streamlit run app.py
```

浏览器会自动打开界面，直接提问即可。

## RAG 工作流程

系统按以下 5 步处理一次提问：

```
① 加载文档    data/ 目录下的 .txt/.md/.pdf → 纯文本
② 切分文本    按 500 字切块，重叠 50 字，切在句子边界上
③ 向量化      Embedding 模型把每个文本块转成向量，写入 FAISS
④ 检索        问题也转成向量，用余弦相似度找最相关的 top_k 块
⑤ 生成        把检索到的条文拼进 Prompt，DeepSeek 基于条文回答
```

其中第 ⑤ 步的 Prompt 做了关键约束：**只依据参考资料回答，资料里没有就直说**，这是抑制幻觉的核心手段。

## 目录结构

```
engineering-qa/
├── app.py              # Streamlit 界面：问答 + 引用展示
├── rag.py              # RAG 核心：加载/切分/向量化/检索/生成
├── requirements.txt    # 依赖
├── .env.example        # 环境变量模板（复制成 .env 填写）
├── .gitignore
└── data/               # 放工程规范文档
```

## 项目背景

作者是土木工程专业背景，正在跨专业学习 AI 应用开发。这个项目把「工程规范知识库」这一土木行业真实需求，和「RAG」这一大模型应用技术结合起来，作为个人从零实现的 RAG 学习与实践作品。

## License

MIT
