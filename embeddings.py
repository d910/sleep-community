# embeddings.py

import dashscope
from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL

dashscope.api_key = DASHSCOPE_API_KEY

class DashScopeEmbeddings:
    """
    通义千问的Embedding包装类
    作用：把文字转成向量
    """
    
    def __init__(self):
        self.model = EMBEDDING_MODEL
    
    def embed_documents(self, texts):
        """
        批量向量化文档
        
        参数:
            texts: 文本列表 ["文本1", "文本2", ...]
        
        返回:
            向量列表 [[0.1,0.2,...], [0.3,0.4,...], ...]
        """
        from dashscope import TextEmbedding
        
        embeddings = []
        for text in texts:
            response = TextEmbedding.call(
                model=self.model,
                input=text
            )
            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                embeddings.append(embedding)
            else:
                # 如果失败，返回空向量
                embeddings.append([0.0] * 1536)
        
        return embeddings
    
    def embed_query(self, text):
        """
        向量化单个查询
        
        参数:
            text: 单个文本字符串
        
        返回:
            向量 [0.1, 0.2, ...]
        """
        from dashscope import TextEmbedding
        
        response = TextEmbedding.call(
            model=self.model,
            input=text
        )
        
        if response.status_code == 200:
            return response.output['embeddings'][0]['embedding']
        else:
            return [0.0] * 1536