# document_processor.py

"""
文档处理模块
作用：加载PDF、切分文档
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        # 文档切分器配置
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # 每块500字符
            chunk_overlap=50,      # 块之间重叠50字符（避免语义断裂）
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
    
    def load_pdf(self, file_path):
        """
        加载单个PDF文件
        
        参数:
            file_path: PDF文件路径
        
        返回:
            文档对象列表
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"✅ 成功加载: {file_path} ({len(documents)}页)")
            return documents
        except Exception as e:
            print(f"❌ 加载失败: {file_path}, 错误: {e}")
            return []
    
    def load_pdf_directory(self, directory_path):
        """
        加载文件夹里的所有PDF
        
        参数:
            directory_path: 文件夹路径
        
        返回:
            所有文档对象列表
        """
        all_documents = []
        
        # 遍历文件夹
        for filename in os.listdir(directory_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                documents = self.load_pdf(file_path)
                all_documents.extend(documents)
        
        print(f"\n📚 总共加载了 {len(all_documents)} 个文档页面")
        return all_documents
    
    def split_documents(self, documents):
        """
        切分文档成小块
        
        参数:
            documents: 文档对象列表
        
        返回:
            切分后的文档块列表
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"✂️ 文档切分完成，共 {len(chunks)} 个块")
        
        # 打印第一个块的示例（调试用）
        if chunks:
            print(f"\n📝 第一个块示例（前100字）：")
            print(chunks[0].page_content[:100] + "...")
        
        return chunks

# 使用示例（你可以测试）
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # 加载data文件夹里的所有PDF
    documents = processor.load_pdf_directory("./data")
    
    # 切分文档
    chunks = processor.split_documents(documents)
    
    print(f"\n✅ 处理完成！")