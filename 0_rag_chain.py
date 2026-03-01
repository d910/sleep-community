# rag_chain.py

"""
RAG检索链模块
作用：实现混合检索 + 生成回答
"""

from langchain_community.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from config import DASHSCOPE_API_KEY, LLM_MODEL

class SleepCommunityRAG:
    """睡眠互助社区RAG系统"""
    
    def __init__(self, official_vectorstore, user_vectorstore):
        """
        初始化RAG系统
        
        参数:
            official_vectorstore: 官方文档向量库
            user_vectorstore: 用户经验向量库
        """
        self.official_store = official_vectorstore
        self.user_store = user_vectorstore
        
        # 初始化大模型
        self.llm = Tongyi(
            model_name=LLM_MODEL,
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=0.7
        )
        
        # 初始化记忆（多轮对话）
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def hybrid_search(self, query, k=3):
        """
        混合检索：同时查询两个向量库
        
        参数:
            query: 用户问题
            k: 每个库返回的结果数
        
        返回:
            {
                "official": [...],  # 官方文档结果
                "user": [...]       # 用户经验结果
            }
        """
        # 从官方文档检索
        official_results = self.official_store.similarity_search(query, k=k)
        
        # 从用户经验检索
        user_results = self.user_store.similarity_search(query, k=k)
        
        return {
            "official": official_results,
            "user": user_results
        }
    
    def format_results(self, results):
        """
        格式化检索结果
        
        参数:
            results: hybrid_search返回的结果
        
        返回:
            格式化后的文本
        """
        formatted = ""
        
        # 格式化官方文档
        formatted += "【权威医学知识】\n"
        for i, doc in enumerate(results["official"]):
            formatted += f"{i+1}. {doc.page_content}\n\n"
        
        # 格式化用户经验
        formatted += "\n【社区用户经验】\n"
        for i, doc in enumerate(results["user"]):
            author = doc.metadata.get("author", "匿名用户")
            helpful = doc.metadata.get("helpful_count", 0)
            formatted += f"{i+1}. {doc.page_content}\n"
            formatted += f"   （分享者：{author}，{helpful}人觉得有用）\n\n"
        
        return formatted
    
    def generate_answer(self, query):
        """
        生成回答（核心方法）
        
        参数:
            query: 用户问题
        
        返回:
            AI回答
        """
        # 1. 混合检索
        results = self.hybrid_search(query, k=3)
        
        # 2. 格式化检索结果
        context = self.format_results(results)
        
        # 3. 构建Prompt
        prompt_template = """
你是一个温柔专业的睡眠互助社区助手「晚安」。

用户提问：{question}

我为你找到了以下信息：

{context}

请基于以上信息，生成一个完整、温柔的回答：

回答要求：
1. 先用1-2句话回应用户的感受
2. 给出专业建议（基于权威医学知识）
3. 分享社区用户的实践经验
4. 语气要温暖、鼓励、有同理心
5. 如果检索结果不够，可以适当补充你的知识，但要标注"补充建议"

回答：
"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "context"]
        )
        
        # 4. 生成回答
        formatted_prompt = prompt.format(question=query, context=context)
        answer = self.llm.invoke(formatted_prompt)
        
        # 5. 保存到记忆
        self.memory.save_context(
            {"question": query},
            {"answer": answer}
        )
        
        return answer, results

# 测试代码
if __name__ == "__main__":
    from vector_store import VectorStoreManager
    
    # 加载向量库
    manager = VectorStoreManager()
    official_store = manager.load_official_vectorstore()
    user_store = manager.load_user_vectorstore()
    
    # 创建RAG系统
    rag = SleepCommunityRAG(official_store, user_store)
    
    # 测试
    print("\n" + "="*50)
    print("测试RAG系统：")
    print("="*50)
    
    query = "我工作压力大，晚上睡不着，怎么办？"
    print(f"\n🙋 用户: {query}")
    
    answer, results = rag.generate_answer(query)
    
    print(f"\n🤖 晚安: {answer}")