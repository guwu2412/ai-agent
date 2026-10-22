import json
from rag_tool import search_resume as _search_resume, init_knowledge_base
from langchain_core.tools import tool
from openai import OpenAI
import requests
from dotenv import load_dotenv
import os


load_dotenv()

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)
def print_banner(title: str, char: str = "="):
    """打印一个漂亮的横幅"""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}")
@tool
def search_resume(query:str):
    """从求职者的简历中检索相关内容。当需要了解求职者的项目经历、技能、教育背景时使用。"""
    return _search_resume(query)
@tool
def generate_question(topic:str, difficulty:str="medium"):
    """根据话题生成一道面试题。当需要向求职者提问时使用。

        Args:
            topic: 面试话题，例如 ReAct Agent、RAG、项目经历
            difficulty: 题目难度，可选 easy/medium/hard，默认 medium
        """
    prompt = f"""你是一位资深技术面试官。请针对以下话题生成一道面试题。
    话题：{topic}
    难度：{difficulty}
    要求：
    1. 问题要具体，不要泛泛而谈
    2. 如果是项目相关，要追问技术细节
    3. 只输出问题本身，不要加任何前缀
    问题："""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()
@tool
def evaluate_answer(question:str, answer:str):
    """评估求职者的回答，给出评分和反馈。当求职者回答完一个问题后使用。

      Args:
          question: 面试官提出的问题
          answer: 求职者的回答内容
      """
    prompt = f"""你是一位技术面试官，请评估以下回答。

    问题：{question}

    求职者回答：{answer}

    请按以下格式输出评估结果（用中文）：

    【评分】X/10
    【亮点】
    - ...
    【不足】
    - ...
    【建议】
    - ...

    评分标准：
    - 准确性（4分）：技术概念是否正确
    - 完整性（3分）：是否覆盖关键点
    - 深度（3分）：是否讲到实现细节、踩坑经验
    """
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content":prompt}],
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()


raw_tools = [search_resume,generate_question,evaluate_answer]
tools_map = {t.name:t for t in raw_tools}
def run_tool(name, args):
   tool_obj = tools_map.get(name)
   if not tool_obj:
       return f"未知工具:{name}"
   try:
       return tool_obj.invoke(args)
   except Exception as e:
       return f"工具执行出错:{str(e)}"

# ---------- 工具 schema（已修复拼写）----------
tools=[
    {
        "type":"function",
        "function":{
            "name":t.name,
            "description":t.description,
            "parameters":t.args_schema.model_json_schema()
        }
    }
    for t in raw_tools
]
# ---------- Agent 支持多轮对话 ----------
def agent(messages, max_steps=5):
    """
    传入完整的消息列表，返回 (最终回答, 更新后的消息列表)
    """
    called_tools = []

    for step in range(max_steps):
        print(f"\n  ⚙️  Agent 思考中... (第 {step + 1}/{max_steps} 轮)")
        response = client.chat.completions.create(
            model="deepseek-chat",        # 改成可用的模型名
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        # 把 assistant 消息加入历史
        messages.append(msg.model_dump(exclude_none=True))

        # 没有工具调用，直接回答
        if not msg.tool_calls:
            if not msg.content or msg.content.strip() == "":
                print("模型返回空内容，强制它直接回答...")
                messages.append({"role": "user", "content": "请直接根据上面的工具结果总结并回答用户问题。"})
                continue
            return msg.content, messages

        # 防重复调用（更安全的做法：先检查，再决定是否加入历史）
        current_call = [(tc.function.name, tc.function.arguments) for tc in msg.tool_calls]
        if called_tools and current_call == called_tools[-1]:
            print("⚠️ 检测到重复调用,强制agent回答")
            # 因为已经把带 tool_calls 的 assistant 消息加进去了，
            # 必须为每个 tool_call 补一个 tool 响应，否则下次请求会报错
            for tc in msg.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "（已忽略重复调用）"
                })
            messages.append({
                "role": "user",
                "content": "你已经调用过这些工具并得到了结果，请直接回答用户的问题，不要再调用工具。"
            })
            continue

        called_tools.append(current_call)

        # 执行工具
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"     🔧 调用工具: {name}")
            print(f"        参数: {args}")
            result = run_tool(name, args)
            # 结果太长时截断，只显示前 100 字
            result_preview = str(result)[:100] + ("..." if len(str(result)) > 100 else "")
            print(f"        结果: {result_preview}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    return "达到最大步数，停止。", messages


if __name__ == "__main__":
    print_banner("🎙️  AI 模拟面试官 启动中", "=")
    print("正在初始化简历库...")
    init_knowledge_base()
    print("✅ 简历库初始化完成")

    messages = [
        {"role": "system", "content": (
            "你是一位资深技术面试官，正在面试一位求职者。\n\n"
            "流程：\n"
            "1. 先用 search_resume 了解简历\n"
            "2. 用 generate_question 出题，一次只问一个问题\n"
            "3. 用户回答后，必须调用 evaluate_answer 评估\n"
            "4. 根据评估结果决定：追问（最多1次）或换下一个话题\n\n"
            "输出要求：\n"
            "- 每次发言不超过 100 字\n"
            "- 一次只问一个问题，不要拆成 1/2/3 点\n"
            "- 追问时直接问，不要重复复述用户的回答\n\n"
            "严格使用工具名：search_resume、generate_question、evaluate_answer"
        )}
    ]

    user_inputs = [
        "面试官你好，我准备好了。",
        "我做过一个手写 ReAct Agent 的项目，用 Python 从零实现了 Function Calling 循环。",
        "防死循环的话，我设置了最大步数限制，还加了重复调用检测。",
    ]

    for i, user_input in enumerate(user_inputs, 1):
        print_banner(f"👤 用户发言 #{i}", "-")
        print(f"  {user_input}")

        messages.append({"role": "user", "content": user_input})
        answer, messages = agent(messages)

        print_banner(f"💬 面试官回复 #{i}", "-")
        print(f"  {answer}")