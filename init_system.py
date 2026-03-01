# init_system.py

"""
系统初始化脚本
第一次运行项目时执行这个文件
"""

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from user_experiences import FAKE_USER_EXPERIENCES

def init_system():
    """初始化系统"""
    
    print("="*60)
    print("🌙 睡眠互助社区 - 系统初始化")
    print("="*60)
    
    # 1. 处理官方文档
    print("\n📚 步骤1：处理官方文档")
    processor = DocumentProcessor()
    documents = processor.load_pdf_directory("./data")
    
    if not documents:
        print("❌ 错误：data文件夹里没有PDF文件！")
        print("💡 请先在data文件夹里放入PDF文档")
        return False
    
    chunks = processor.split_documents(documents)
    
    # 2. 创建向量库
    print("\n🔄 步骤2：创建向量数据库")
    manager = VectorStoreManager()
    
    official_store = manager.create_official_vectorstore(chunks)
    user_store = manager.create_user_vectorstore(FAKE_USER_EXPERIENCES)
    
    # 3. 测试检索
    print("\n🔍 步骤3：测试检索功能")
    test_query = "我睡不着怎么办？"
    print(f"测试查询：{test_query}")
    
    official_results = manager.search_similar(official_store, test_query, k=2)
    print(f"\n【官方文档】找到 {len(official_results)} 条结果")
    
    user_results = manager.search_similar(user_store, test_query, k=2)
    print(f"【用户经验】找到 {len(user_results)} 条结果")
    
    print("\n" + "="*60)
    print("✅ 系统初始化完成！")
    print("="*60)
    print("\n💡 现在可以运行: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    init_system()