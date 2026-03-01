# app.py

"""
睡眠互助社区 - 主程序
美观的Streamlit界面
"""

import streamlit as st
from vector_store import VectorStoreManager
from rag_chain import SleepCommunityRAG
from user_experiences import FAKE_USER_EXPERIENCES
from document_processor import DocumentProcessor
import os

# ===== 页面配置（必须放在最前面）=====
st.set_page_config(
    page_title="睡眠互助社区",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 自定义CSS样式（美化界面）=====
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* 副标题样式 */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* 消息框样式 */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .ai-message {
        background: #f7f7f7;
        color: #333333;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 结果卡片样式 */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* 标签样式 */
    .tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    
    /* 统计数字样式 */
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* 隐藏Streamlit默认的菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===== 初始化session_state =====
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.rag_system = None
    st.session_state.chat_history = []

# ===== 侧边栏 =====
with st.sidebar:
    st.markdown("### 🌙 睡眠互助社区")
    st.markdown("---")
    
    # 系统状态
    if st.session_state.initialized:
        st.success("✅ 系统已就绪")
    else:
        st.warning("⚠️ 系统未初始化")
    
    st.markdown("---")
    
    # 功能导航
    page = st.radio(
        "选择功能",
        ["💬 智能问答", "📝 分享经验", "📊 社区统计"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # 初始化按钮
    if st.button("🔄 初始化系统", use_container_width=True):
        with st.spinner("正在初始化向量数据库..."):
            try:
                # 检查是否已有数据库
                if os.path.exists("./chroma_db"):
                    st.info("检测到已有数据库，加载中...")
                    manager = VectorStoreManager()
                    official_store = manager.load_official_vectorstore()
                    user_store = manager.load_user_vectorstore()
                else:
                    st.info("首次运行，创建向量数据库...")
                    
                    # 1. 处理官方文档
                    processor = DocumentProcessor()
                    documents = processor.load_pdf_directory("./data")
                    chunks = processor.split_documents(documents)
                    
                    # 2. 创建向量库
                    manager = VectorStoreManager()
                    official_store = manager.create_official_vectorstore(chunks)
                    user_store = manager.create_user_vectorstore(FAKE_USER_EXPERIENCES)
                
                # 3. 创建RAG系统
                rag_system = SleepCommunityRAG(official_store, user_store)
                
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                
                st.success("✅ 初始化完成！")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 初始化失败：{str(e)}")
    
    # 清空对话历史
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ===== 主页面 =====

# 页面标题
st.markdown('<h1 class="main-title">🌙 睡眠互助社区</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">专业知识 × 真实经验 | AI驱动的失眠者知识库</p>', unsafe_allow_html=True)

# 根据选择的功能显示不同页面
if page == "💬 智能问答":
    # ===== 智能问答页面 =====
    
    if not st.session_state.initialized:
        st.warning("⚠️ 请先在左侧初始化系统")
    else:
        # 输入框
        st.markdown("### 💬 向社区提问")
        user_question = st.text_input(
            "描述你的睡眠困扰...",
            placeholder="比如：我最近工作压力大，晚上翻来覆去睡不着...",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            ask_button = st.button("🔍 提问", use_container_width=True, type="primary")
        with col2:
            if st.button("💡 随机示例", use_container_width=True):
                examples = [
                    "我工作压力大，晚上睡不着怎么办？",
                    "半夜总是醒来，怎么改善？",
                    "褪黑素应该怎么吃？",
                    "睡前做什么能更容易入睡？"
                ]
                import random
                user_question = random.choice(examples)
                st.rerun()
        
        # 处理提问
        if ask_button and user_question:
            with st.spinner("🤔 AI正在思考..."):
                try:
                    # 生成回答
                    answer, results = st.session_state.rag_system.generate_answer(user_question)
                    
                    # 保存到对话历史
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer,
                        "results": results
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 生成失败：{str(e)}")
        
        # 显示对话历史
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💭 对话历史")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                # 用户问题
                st.markdown(f'<div class="user-message">🙋 {chat["question"]}</div>', 
                          unsafe_allow_html=True)
                
                # AI回答
                st.markdown(f'<div class="ai-message">🤖 晚安：<br><br>{chat["answer"]}</div>', 
                          unsafe_allow_html=True)
                
                # 展开查看检索结果
                with st.expander("📚 查看检索到的相关内容"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📖 权威医学知识")
                        for j, doc in enumerate(chat["results"]["official"]):
                            st.markdown(f"**{j+1}.** {doc.page_content[:200]}...")
                    
                    with col2:
                        st.markdown("#### 💡 社区用户经验")
                        for j, doc in enumerate(chat["results"]["user"]):
                            author = doc.metadata.get("author", "匿名")
                            helpful = doc.metadata.get("helpful_count", 0)
                            st.markdown(f"**{j+1}.** {doc.page_content}")
                            st.caption(f"👤 {author} · 👍 {helpful}人觉得有用")
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")

elif page == "📝 分享经验":
    # ===== 分享经验页面 =====
    
    st.markdown("### 📝 分享你的助眠经验")
    st.info("💡 你的经验可能帮助其他失眠者！分享后将被存入社区知识库。")
    
    with st.form("share_experience"):
        experience_content = st.text_area(
            "描述你的助眠方法或经验",
            placeholder="比如：我发现每晚听白噪音能让我更快入睡...",
            height=150
        )
        
        col1, col2 = st.columns(2)
        with col1:
            author_name = st.text_input("昵称（可选）", placeholder="匿名用户")
        with col2:
            tags = st.multiselect(
                "相关标签",
                ["入睡困难", "中途觉醒", "早醒", "焦虑", "压力", "运动", "饮食", "冥想", "音乐", "其他"]
            )
        
        submit_button = st.form_submit_button("✅ 提交分享", use_container_width=True, type="primary")
        
        if submit_button:
            if experience_content:
                with st.spinner("🔍 AI正在审核内容..."):
                    # 这里可以加AI审核逻辑（MVP版本先跳过）
                    import time
                    time.sleep(1)
                    
                    st.success("✅ 提交成功！感谢你的分享！")
                    st.balloons()
                    
                    # TODO: 实际保存到向量数据库
                    st.info("📝 你的经验已被记录，将在下次系统初始化时生效")
            else:
                st.warning("⚠️ 请填写经验内容")

elif page == "📊 社区统计":
    # ===== 社区统计页面 =====
    
    st.markdown("### 📊 社区数据统计")
    
    # 模拟统计数据
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-number">156</div>', unsafe_allow_html=True)
        st.markdown("权威医学文档")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-number">10</div>', unsafe_allow_html=True)
        st.markdown("用户经验分享")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-number">342</div>', unsafe_allow_html=True)
        st.markdown("累计提问")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="stat-number">89%</div>', unsafe_allow_html=True)
        st.markdown("问题解决率")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 热门话题
    st.markdown("### 🔥 热门话题")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 最常问的问题")
        topics = [
            ("入睡困难", 89),
            ("中途觉醒", 67),
            ("焦虑失眠", 54),
            ("褪黑素使用", 43),
            ("睡眠质量", 38)
        ]
        for topic, count in topics:
            st.progress(count/100)
            st.caption(f"**{topic}** - {count}次提问")
    
    with col2:
        st.markdown("#### 最受欢迎的经验")
        experiences = [
            ("听白噪音助眠", 45),
            ("睡前冥想", 41),
            ("调整作息时间", 37),
            ("睡前泡脚", 28),
            ("减少咖啡因", 22)
        ]
        for exp, likes in experiences:
            st.progress(likes/50)
            st.caption(f"**{exp}** - {likes}人觉得有用")

# ===== 页面底部 =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9rem; padding: 2rem 0;'>
    <p>🌙 睡眠互助社区 | Powered by LangChain + Chroma + 通义千问</p>
    <p>💡 专业知识来自医学文档 · 用户经验来自社区分享</p>
    <p style='font-size: 0.8rem; color: #ccc;'>免责声明：本平台提供的建议仅供参考，不能替代专业医疗建议。严重失眠请就医。</p>
</div>
""", unsafe_allow_html=True)