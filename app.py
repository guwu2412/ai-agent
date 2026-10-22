import streamlit as st
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import time
import uuid

from proj2 import build_graph, init_knowledge_base

load_dotenv()

# ========================
# 1. 页面配置 & 纯白极简风 CSS
# ========================
st.set_page_config(page_title="AI 面试官工作台", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    /* 1. 隐藏默认菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 2. 全局背景：纯白色，护眼清爽 */
    .stApp {
        background-color: #FFFFFF;
    }

    /* 3. 侧边栏：极浅的灰色，区分主次 */
    [data-testid="stSidebar"] {
        background-color: #F7F7F9;
        border-right: 1px solid #E5E5E5;
    }

    /* 4. 全局文字颜色：深灰，保证高对比度 */
    h1, h2, h3, h4, p, span, div, li {
        color: #2D3748 !important;
    }

    /* 5. 聊天气泡容器：去掉默认丑陋的边框 */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 16px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* 6. 用户气泡：优雅的浅蓝色 */
    [data-testid="stChatMessage"]:has(div[aria-label="user avatar"]) {
        background-color: #E6F2FF;
        border: 1px solid #D0E3F9;
    }

    /* 7. AI 气泡：带边框的纯白 */
    [data-testid="stChatMessage"]:has(div[aria-label="assistant avatar"]) {
        background-color: #FFFFFF;
        border: 1px solid #E5E5E5;
    }

    /* 8. 底部输入框美化：白底、细边框、阴影 */
    [data-testid="stChatInput"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important; 
        color: #2D3748 !important; 
        border: 1px solid #D1D5DB !important; 
        border-radius: 24px !important; /* 更圆润 */
        box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
        padding: 12px 20px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #9CA3AF !important; 
    }

    /* 9. 给底部留出安全距离，防止输入框被切掉 */
    .block-container {
        padding-bottom: 120px !important; 
    }

    /* 10. 侧边栏按钮统一圆角 */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ========================
# 2. 初始化
# ========================
@st.cache_resource
def setup():
    init_knowledge_base()
    return build_graph()


app = setup()

# ========================
# 3. 状态管理
# ========================
if "thread_id" in st.query_params:
    st.session_state.thread_id = st.query_params["thread_id"]
elif "thread_id" not in st.session_state:
    new_id = f"web_{uuid.uuid4().hex[:8]}"
    st.session_state.thread_id = new_id
    st.query_params["thread_id"] = new_id

config = {"configurable": {"thread_id": st.session_state.thread_id}}

if "messages" not in st.session_state:
    st.session_state.messages = []
    try:
        graph_state = app.get_state(config)
        if graph_state and graph_state.values:
            for msg in graph_state.values.get("messages", []):
                if msg.type == "human" and msg.content:
                    st.session_state.messages.append({"role": "user", "content": msg.content})
                elif msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
                    st.session_state.messages.append({"role": "assistant", "content": msg.content})
    except Exception:
        pass

if "is_first_turn" not in st.session_state: st.session_state.is_first_turn = True
if "interview_ended" not in st.session_state: st.session_state.interview_ended = False
if "report" not in st.session_state: st.session_state.report = ""

# ========================
# 4. 侧边栏控制台
# ========================
with st.sidebar:
    st.title("🎛️ 控制台")
    st.info(f"会话 ID: {st.session_state.thread_id}")
    st.divider()

    if not st.session_state.interview_ended:
        # 改用 Streamlit 原生的红色主按钮
        if st.button("🛑 提前结束面试", type="primary", use_container_width=True):
            with st.spinner("结算成绩中..."):
                init_state = {"messages": [HumanMessage(content="我决定结束本次面试，请立刻生成最终评估报告。")],
                              "topic_index": 999, "follow_up_count": 0}
                result = app.invoke(init_state, config=config)

                final_report = "报告生成失败。"
                for m in reversed(result["messages"]):
                    if m.type == "ai" and not getattr(m, "tool_calls", None) and m.content:
                        final_report = m.content
                        break

                st.session_state.interview_ended = True
                st.session_state.report = final_report
                st.session_state.messages.append({"role": "assistant", "content": final_report})
                st.rerun()

    if st.button("🔄 开启新面试", use_container_width=True):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

# ========================
# 5. 主界面布局
# ========================
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("<h2 style='text-align: center; color: #1A202C;'>🎙️ AI 技术面试官</h2>", unsafe_allow_html=True)
                unsafe_allow_html=True)
    st.divider()

    # 渲染聊天记录
    for msg in st.session_state.messages:
        avatar = "👨‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # 输入框与处理逻辑
    if not st.session_state.interview_ended:
        if user_input := st.chat_input("在此输入你的回答..."):
            if user_input.upper() == 'YL':
                st.session_state.interview_ended = True
                st.rerun()

            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="👨‍💻"):
                st.markdown(user_input)

            if st.session_state.is_first_turn:
                init_state = {"messages": [HumanMessage(content=user_input)], "topic_index": 0, "follow_up_count": 0}
                st.session_state.is_first_turn = False
            else:
                init_state = {"messages": [HumanMessage(content=user_input)]}

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("思考中..."):
                    result = app.invoke(init_state, config=config)

                    final_content = "（思考中...）"
                    for m in reversed(result["messages"]):
                        if m.type == "ai" and not getattr(m, "tool_calls", None) and m.content:
                            final_content = m.content
                            break

                    last_msg = final_content
                    if "综合评分" in last_msg:
                        st.session_state.interview_ended = True
                        st.session_state.report = last_msg

                # 打字机效果
                message_placeholder = st.empty()
                full_response = ""
                for chunk in last_msg.split():
                    full_response += chunk + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": last_msg})

# 右侧：报告面板
with col2:
    st.markdown("<h3 style='color: #1A202C;'>📋 面试评估报告</h3>", unsafe_allow_html=True)
    if st.session_state.get("interview_ended") and st.session_state.get("report"):
        st.success("✅ 面试已结束")
        with st.expander("点击查看完整报告", expanded=True):
            st.markdown(st.session_state.report)

        st.download_button(
            label="⬇️ 下载完整报告",
            data=st.session_state.report,
            file_name="interview_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        # 使用浅色提示框
        st.info("正在面试中... 结束后将在此展示评估报告。")