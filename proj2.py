import os
import sqlite3
import logging
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from rag_tool import search_resume as _search_resume, init_knowledge_base

load_dotenv()

# ========================
# 专业日志配置
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("interview.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ========================
# 状态定义
# ========================
class Astate(TypedDict):
    messages: Annotated[list, add_messages]
    topic_index: int
    follow_up_count: int


TOPICS = [
    "项目经历：请介绍你简历里最有代表性的项目",
    "RAG：请讲讲你简历里的知识库问答系统",
    "工程能力：FastAPI、Docker、日志、评估",
    "系统设计：设计一个支持 1000 QPS 的 Agent 服务",
    "行为问题：团队协作或抗压能力",
]


# ========================
# 工具与模型
# ========================
@tool
def search_resume(query: str) -> str:
    """从求职者的简历中检索相关内容。"""
    return _search_resume(query)


tools = [search_resume]
tools_map = {t.name: t for t in tools}

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    temperature=0.3,
).bind_tools(tools)

SYSTEM_PROMPT = """你是一位资深技术面试官。请遵守以下规则：
1. 每次只问一个问题，不超过 100 字。
2. 不要复述用户的回答，保持专业。
3. 基于用户的回答进行追问。
4. 严禁使用"你提到xxx"的句式编造候选人没说过的话。
5. 【情商规则】如果候选人回答“不知道”、“没做过”或明显敷衍，请直接指出并切换到下一个话题，不要强行追问。
"""


# ========================
# 节点 1：调用模型（清洗超长历史，强制指令）
# ========================
def call_model(state: Astate):
    messages = state["messages"]
    topic_index = state.get("topic_index", 0)
    current_topic = TOPICS[topic_index] if topic_index < len(TOPICS) else "面试即将结束"

    topic_hint = SystemMessage(content=(
        f"【系统强制指令】上一个话题已经彻底作废！\n"
        f"绝对禁止再提及任何关于之前项目的细节！\n"
        f"你现在必须立刻向候选人提出一个关于该话题的全新问题：“{current_topic}”。\n"
    ))

    # 长文本折叠：防止候选人写小作文导致模型注意力涣散
    cleaned_messages = []
    for m in messages:
        if m.type == "human" and len(m.content) > 150:
            cleaned_messages.append(HumanMessage(content="（候选人刚刚回答了一大段详细的技术方案，细节已省略）"))
        else:
            cleaned_messages.append(m)

    if not cleaned_messages or cleaned_messages[0].type != "system":
        final_messages = [SystemMessage(content=SYSTEM_PROMPT), topic_hint] + cleaned_messages
    else:
        final_messages = [cleaned_messages[0], topic_hint] + [m for m in cleaned_messages[1:] if m.type != "system"]

    response = llm.invoke(final_messages)
    return {"messages": [response]}


# ========================
# 节点 2：执行工具
# ========================
def execute_tool(state: Astate):
    last_message = state["messages"][-1]
    tool_messages = []
    for tc in last_message.tool_calls:
        name = tc["name"]
        args = tc["args"]
        logger.info(f"🔧 调用工具: {name} | 参数: {args}")
        tool_obj = tools_map.get(name)
        if not tool_obj:
            result = f"未知工具:{name}"
        else:
            try:
                raw = tool_obj.invoke(args)
                result = str(raw.content) if hasattr(raw, "content") else str(raw)
            except Exception as e:
                result = f"工具执行出错: {str(e)}"

        logger.info(f"✅ 工具结果: {result[:80]}...")
        tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": tool_messages}


# ========================
# 节点 3：控制话题切换（防敷衍卡死）
# ========================
def check_next_topic(state: Astate):
    messages = state["messages"]
    last_answer = next((m.content for m in reversed(messages) if m.type == "human"), "")
    follow_up = state.get("follow_up_count", 0)
    topic_index = state.get("topic_index", 0)

    # 统计最近 6 条消息中有几条是极短回复（比如“好的”、“嗯”）
    short_replies = sum(1 for m in messages[-6:] if m.type == "human" and len(m.content.strip()) < 10)
    if short_replies >= 3:
        logger.info("⚠️ 检测到连续敷衍/极短回复，强制切换话题")
        return {"topic_index": topic_index + 1, "follow_up_count": 0}

    # 第一次寒暄极短，允许
    if len(last_answer.strip()) < 10 and follow_up == 0:
        return {}

    # 回答简短且没追问过，允许追问 1 次
    if len(last_answer.strip()) < 50 and follow_up == 0:
        return {"follow_up_count": 1}

    # 正常回答，直接切话题
    logger.info(f"🔄 切换至下一话题 (Topic {topic_index + 1})")
    return {"topic_index": topic_index + 1, "follow_up_count": 0}


# ========================
# 节点 4：生成最终报告
# ========================
def generate_report(state: Astate):
    messages = state["messages"]
    history_text = "".join(f"{'求职者' if m.type == 'human' else '面试官'}：{m.content}\n" for m in messages if
                           m.type in ["human", "ai"] and m.content)

    prompt = f"请根据以下面试记录，生成一份综合评估报告：\n【面试记录】\n{history_text}\n\n格式要求：\n【综合评分】X/100\n【总体评价】\n【亮点】\n【不足】\n【改进建议】\n"
    report = llm.invoke([HumanMessage(content=prompt)]).content
    logger.info("📋 最终面试报告已生成。")
    return {"messages": [AIMessage(content=report)]}


# ========================
# 路由与图构建
# ========================
def should_continue(state: Astate):
    last_message = state["messages"][-1]
    return "tools" if hasattr(last_message, "tool_calls") and last_message.tool_calls else "next"


def route_after_check(state: Astate):
    return "report" if state.get("topic_index", 0) >= len(TOPICS) else "end"


def build_graph():
    graph = StateGraph(Astate)
    graph.add_node("call_model", call_model)
    graph.add_node("execute_tools", execute_tool)
    graph.add_node("check_next_topic", check_next_topic)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("call_model")
    graph.add_conditional_edges("call_model", should_continue, {"tools": "execute_tools", "next": "check_next_topic"})
    graph.add_edge("execute_tools", "call_model")
    graph.add_conditional_edges("check_next_topic", route_after_check, {"report": "generate_report", "end": END})
    graph.add_edge("generate_report", END)

    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    return graph.compile(checkpointer=memory)