import os
from dotenv import load_dotenv

load_dotenv()  # 必须先加载环境变量

import chromadb
from openai import OpenAI

# 1. 初始化 Embedding 客户端（使用硅基流动的 API）
embed_client = OpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)


def get_embedding(text: str):
    """将文本转为向量"""
    resp = embed_client.embeddings.create(
        model="BAAI/bge-m3",
        input=text
    )
    return resp.data[0].embedding


def init_knowledge_base():
    """初始化知识库，按标题切块"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="resume")

    if collection.count() > 0:
        print(f"简历库已存在，共 {collection.count()} 条记录，跳过重建。")
        return collection

    with open("resume.txt", "r", encoding="utf-8") as f:
        content = f.read()
    #导入正则模块
    import re
    # 同时支持【标题】和 Markdown 的 ## / ### 标题
    pattern = r"(?=【)|(?=^#{2,3}\s)"
    parts = re.split(pattern, content, flags=re.MULTILINE)

    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) < 20:  # 太短的跳过
            continue
        chunks.append(part)

    if not chunks:
        raw = [p.strip() for p in content.split("\n\n") if p.strip()]
        chunks = [p for p in raw if len(p) >= 20]

    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        collection.add(
            embeddings=[vector],
            documents=[chunk],
            ids=[f"id_{i}"]
        )
    print(f"简历库构建完成，共 {len(chunks)} 个语义块。")
    return collection

def search_resume(query: str) -> str:
    """RAG 检索工具（给 Agent 调用的）"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="resume")

        # 1. 获取用户问题的向量
        query_vector = get_embedding(query)

        # 2. 相似度检索，取前 3 条
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )

        docs = results["documents"][0]
        if not docs:
            return "简历库中没有找到相关内容。"

        # 3. 把最相关的片段拼成字符串返回给模型
        return "\n".join(docs)
    except Exception as e:
        return f"知识库检索失败：{str(e)}"