# rag_chain.py (新版 - 使用LCEL)

"""
RAG检索链模块（LangChain 1.x 新架构）
作用：使用LCEL实现混合检索 + 生成回答
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

# ParalleLambda
from langchain_core.runnables import RunnableLambda


from langchain_community.llms import Tongyi
from config import DASHSCOPE_API_KEY, LLM_MODEL

class SleepCommunityRAG:
    """睡眠互助社区RAG系统（新版LCEL）"""
    
    def __init__(self, official_vectorstore, user_vectorstore):
        self.official_store = official_vectorstore
        self.user_store = user_vectorstore
        
        # 初始化大模型
        self.llm = Tongyi(
            model_name=LLM_MODEL,
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=0.7
        )
        
        # 创建LCEL链
        self.chain = self._create_chain()
        
        # 对话历史（简单版，存在内存里）
        self.chat_history = []
    
    def _create_chain(self):
        """创建LCEL链"""
        
        # 1. 定义Prompt模板
        prompt = ChatPromptTemplate.from_template("""
你是温柔专业的睡眠互助社区助手「晚安」。

用户提问：{question}

我为你找到了以下信息：

【权威医学知识】
{official_context}

【社区用户经验】
{user_context}

请基于以上信息，生成一个完整、温柔的回答：

回答要求：
1. 先用1-2句话回应用户的感受
2. 给出专业建议（基于权威医学知识）
3. 分享社区用户的实践经验
4. 语气要温暖、鼓励、有同理心
5. 如果检索结果不够，可以适当补充你的知识，但要标注"补充建议"

回答：
""")
        
        # 2. 定义检索函数（RunnableLambda）
        def retrieve_and_format(inputs):
            """混合检索并格式化"""
            question = inputs["question"]
            
            # 从两个向量库检索
            official_docs = self.official_store.similarity_search(question, k=3)
            user_docs = self.user_store.similarity_search(question, k=3)
            
            # 格式化官方文档
            official_context = "\n\n".join([
                f"{i+1}. {doc.page_content}" 
                for i, doc in enumerate(official_docs)
            ])
            
            # 格式化用户经验
            user_context = "\n\n".join([
                f"{i+1}. {doc.page_content}\n   （分享者：{doc.metadata.get('author', '匿名')}，{doc.metadata.get('helpful_count', 0)}人觉得有用）"
                for i, doc in enumerate(user_docs)
            ])
            
            return {
                "question": question,
                "official_context": official_context,
                "user_context": user_context
            }
        
        # 3. 创建LCEL链
        chain = (
            RunnableLambda(retrieve_and_format)  # 检索+格式化
            | prompt                              # 生成Prompt
            | self.llm                            # 调用大模型
            | StrOutputParser()                   # 解析输出
        )
        
        return chain
    
    def generate_answer(self, query):
        """
        生成回答（主方法）
        
        参数:
            query: 用户问题
        
        返回:
            (answer, results) - 回答和检索结果
        """
        # 1. 调用LCEL链
        answer = self.chain.invoke({"question": query})
        
        # 2. 获取检索结果（用于展示）
        official_results = self.official_store.similarity_search(query, k=3)
        user_results = self.user_store.similarity_search(query, k=3)
        
        results = {
            "official": official_results,
            "user": user_results
        }
        
        # 3. 保存对话历史
        self.chat_history.append({
            "question": query,
            "answer": answer
        })
        
        return answer, results
    
    def stream_answer(self, query):
        """流式生成回答（演示用）"""
        for chunk in self.chain.stream({"question": query}):
            yield chunk

# 测试代码
if __name__ == "__main__":
    from vector_store import VectorStoreManager
    
    manager = VectorStoreManager()
    official_store = manager.load_official_vectorstore()
    user_store = manager.load_user_vectorstore()
    
    rag = SleepCommunityRAG(official_store, user_store)
    
    query = "我工作压力大，晚上睡不着，怎么办？"
    print(f"\n🙋 用户: {query}")
    
    answer, results = rag.generate_answer(query)
    print(f"\n🤖 晚安: {answer}")