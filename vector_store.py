# vector_store.py

"""
向量数据库管理模块
作用：创建、查询Chroma数据库
"""

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from embeddings import DashScopeEmbeddings
from config import CHROMA_DB_PATH, COLLECTION_NAME_OFFICIAL, COLLECTION_NAME_USER

class VectorStoreManager:
    """向量数据库管理器"""
    
    def __init__(self):
        self.embeddings = DashScopeEmbeddings()
        self.db_path = CHROMA_DB_PATH
    
    def create_official_vectorstore(self, documents):
        """
        创建官方文档的向量库
        
        参数:
            documents: 文档块列表
        
        返回:
            Chroma向量库对象
        """
        print(f"\n🔄 正在创建官方文档向量库...")
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME_OFFICIAL,
            persist_directory=self.db_path
        )
        
        print(f"✅ 官方文档向量库创建完成！（{len(documents)}个文档块）")
        return vectorstore
    
    def create_user_vectorstore(self, user_experiences):
        """
        创建用户经验的向量库
        
        参数:
            user_experiences: 用户经验列表
                [
                    {
                        "content": "...",
                        "author": "...",
                        "helpful_count": 23
                    },
                    ...
                ]
        
        返回:
            Chroma向量库对象
        """
        print(f"\n🔄 正在创建用户经验向量库...")
        
        # 把用户经验转成LangChain的Document格式
        from langchain_core.documents import Document
        
        documents = []
        for exp in user_experiences:
            doc = Document(
                page_content=exp["content"],
                metadata={
                    "source": "user",
                    "author": exp["author"],
                    "helpful_count": exp["helpful_count"],
                    "tags": exp.get("tags", [])
                }
            )
            documents.append(doc)
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME_USER,
            persist_directory=self.db_path
        )
        
        print(f"✅ 用户经验向量库创建完成！（{len(documents)}条经验）")
        return vectorstore
    
    def load_official_vectorstore(self):
        """
        加载已存在的官方文档向量库
        
        返回:
            Chroma向量库对象
        """
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME_OFFICIAL,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )
        return vectorstore
    
    def load_user_vectorstore(self):
        """
        加载已存在的用户经验向量库
        
        返回:
            Chroma向量库对象
        """
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME_USER,
            embedding_function=self.embeddings,
            persist_directory=self.db_path
        )
        return vectorstore
    
    def search_similar(self, vectorstore, query, k=3):
        """
        搜索相似内容
        
        参数:
            vectorstore: 向量库对象
            query: 查询文本
            k: 返回结果数量
        
        返回:
            相似文档列表
        """
        results = vectorstore.similarity_search(query, k=k)
        return results

# 测试代码
if __name__ == "__main__":
    from user_experiences import FAKE_USER_EXPERIENCES
    from document_processor import DocumentProcessor
    
    # 1. 处理官方文档
    processor = DocumentProcessor()
    documents = processor.load_pdf_directory("./data")
    chunks = processor.split_documents(documents)
    
    # 2. 创建向量库
    manager = VectorStoreManager()
    
    # 创建官方文档向量库
    official_store = manager.create_official_vectorstore(chunks)
    
    # 创建用户经验向量库
    user_store = manager.create_user_vectorstore(FAKE_USER_EXPERIENCES)
    
    # 3. 测试搜索
    print("\n" + "="*50)
    print("测试搜索功能：")
    print("="*50)
    
    query = "我睡不着怎么办？"
    print(f"\n🔍 查询: {query}")
    
    print("\n【官方文档结果】")
    official_results = manager.search_similar(official_store, query, k=2)
    for i, doc in enumerate(official_results):
        print(f"\n{i+1}. {doc.page_content[:100]}...")
    
    print("\n【用户经验结果】")
    user_results = manager.search_similar(user_store, query, k=2)
    for i, doc in enumerate(user_results):
        print(f"\n{i+1}. {doc.page_content}")
        print(f"   作者: {doc.metadata['author']}")
        print(f"   有用: {doc.metadata['helpful_count']}人")