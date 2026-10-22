from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from proj2 import build_graph, init_knowledge_base
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
app = FastAPI(
    title="AI 面试官 API",
    description="基于 LangGraph + RAG 的智能面试官后端服务",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 替换默认的 /docs 路由，使用国内 CDN
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.15.5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.15.5/swagger-ui.css",
    )
# 在服务启动时，初始化知识库和引擎
print("正在初始化后端引擎...")
init_knowledge_base()
graph_app = build_graph()


# 定义请求体结构
class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    is_finished: bool


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """处理用户输入，返回 AI 回复"""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}

        # 判断是不是第一轮（如果没有历史记录的话）
        state = graph_app.get_state(config)
        if not state or not state.values:
            init_state = {"messages": [HumanMessage(content=request.message)], "topic_index": 0, "follow_up_count": 0}
        else:
            init_state = {"messages": [HumanMessage(content=request.message)]}

        # 执行 LangGraph
        result = graph_app.invoke(init_state, config=config)

        # 提取真正的最终回答（过滤掉工具调用中间态）
        final_content = "（处理中...）"
        for m in reversed(result["messages"]):
            if m.type == "ai" and not getattr(m, "tool_calls", None) and m.content:
                final_content = m.content
                break

        # 判断面试是否结束（如果回答里包含综合评分，说明报告出了）
        is_finished = "综合评分" in final_content

        return ChatResponse(
            thread_id=request.thread_id,
            reply=final_content,
            is_finished=is_finished
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history/{thread_id}")
def get_history(thread_id: str):
    """获取某个用户的历史对话"""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph_app.get_state(config)
        if not state or not state.values:
            return {"messages": []}

        msgs = []
        for m in state.values.get("messages", []):
            if m.type == "human" and m.content:
                msgs.append({"role": "user", "content": m.content})
            elif m.type == "ai" and m.content and not getattr(m, "tool_calls", None):
                msgs.append({"role": "assistant", "content": m.content})
        return {"messages": msgs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # 启动 FastAPI 服务
    uvicorn.run(app, host="0.0.0.0", port=8000)