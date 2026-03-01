# config.py

#通义千问API key
DASHSCOPE_API_KEY = st.secrets["API_KEY"]

# Embedding模型配置
EMBEDDING_MODEL = "text-embedding-v1"

# 大模型配置
LLM_MODEL = "qwen-turbo"

# chroma数据库配置
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME_OFFICIAL = "official_docs"
COLLECTION_NAME_USER = "user_experiences"