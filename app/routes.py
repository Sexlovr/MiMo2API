"""API路由"""

import time
import uuid
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from .models import (
    OpenAIRequest, OpenAIResponse, OpenAIChoice, OpenAIMessage,
    OpenAIDelta, OpenAIUsage, ParseCurlRequest, TestAccountRequest
)
from .config import config_manager, MimoAccount
from .mimo_client import MimoClient
from .utils import parse_curl, build_query_from_messages

router = APIRouter()


def validate_api_key(authorization: Optional[str]) -> bool:
    """验证API Key"""
    if not authorization:
        return False

    # 移除"Bearer "前缀
    key = authorization.replace("Bearer ", "").strip()
    return config_manager.validate_api_key(key)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIRequest,
    authorization: Optional[str] = Header(None)
):
    """OpenAI兼容的聊天接口"""

    # 验证API Key
    if not validate_api_key(authorization):
        raise HTTPException(status_code=401, detail={"error": {"message": "invalid api key"}})

    # 获取下一个Mimo账号
    account = config_manager.get_next_account()
    if not account:
        raise HTTPException(status_code=503, detail={"error": {"message": "no mimo account"}})

    # 构建查询字符串和配置
    query, thinking, search = build_query_from_messages(request.messages)

    # 如果请求中明确指定了 reasoning_effort，则覆盖标签中的 thinking 设置
    if request.reasoning_effort:
        thinking = True

    # 处理会话ID，实现会话持续
    # 如果 OpenAI 请求中有 user 字段，我们将其作为 conversationId
    # 否则，我们根据消息内容生成一个稳定的 conversationId，以防 session 丢失
    import hashlib
    conversation_id = request.user if hasattr(request, 'user') and request.user else None
    if not conversation_id and len(request.messages) > 0:
        # 使用第一条消息的内容作为 Session 标识符（简单实现）
        first_msg = request.messages[0].content[:100]
        conversation_id = hashlib.md5(first_msg.encode()).hexdigest()

    # 创建Mimo客户端
    client = MimoClient(account)

    # 流式响应
    if request.stream:
        return StreamingResponse(
            stream_response(client, query, thinking, request.model, conversation_id, search),
            media_type="text/event-stream"
        )

    # 非流式响应
    try:
        content, think_content, usage = await client.call_api(query, thinking, request.model, conversation_id, search)

        # 如果有思考内容，拼接到回复前面
        full_content = content
        if think_content:
            full_content = f"<think>{think_content}</think>\n{content}"

        response = OpenAIResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=full_content),
                    finish_reason="stop"
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=usage.get("promptTokens", 0),
                completion_tokens=usage.get("completionTokens", 0),
                total_tokens=usage.get("promptTokens", 0) + usage.get("completionTokens", 0)
            )
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": {"message": str(e)}})


async def stream_response(client: MimoClient, query: str, thinking: bool, model: str, conversation_id: str = None, search: bool = False):
    """流式响应生成器"""

    msg_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # 发送初始role delta
    yield f"data: {json.dumps(OpenAIResponse(id=msg_id, object='chat.completion.chunk', created=int(time.time()), model=model, choices=[OpenAIChoice(index=0, delta=OpenAIDelta(role='assistant'))]).dict())}\n\n"

    buffer = ""
    in_think = False

    try:
        async for sse_data in client.stream_api(query, thinking, model, conversation_id, search):
            content = sse_data.get("content", "")
            if not content:
                continue

            buffer += content
            text = buffer.replace("\x00", "")

            # 处理<think>标签
            while True:
                if not in_think:
                    # 查找<think>标签
                    idx = text.find("<think>")
                    if idx != -1:
                        # 发送<think>之前的内容
                        if idx > 0:
                            chunk = OpenAIResponse(
                                id=msg_id,
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=model,
                                choices=[OpenAIChoice(index=0, delta=OpenAIDelta(content=text[:idx]))]
                            )
                            yield f"data: {json.dumps(chunk.dict())}\n\n"

                        in_think = True
                        text = text[idx + 7:]
                        continue

                    # 保留最后7个字符以防<think>被截断
                    safe = len(text) - 7
                    if safe > 0:
                        chunk = OpenAIResponse(
                            id=msg_id,
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=model,
                            choices=[OpenAIChoice(index=0, delta=OpenAIDelta(content=text[:safe]))]
                        )
                        yield f"data: {json.dumps(chunk.dict())}\n\n"
                        text = text[safe:]
                    break

                else:
                    # 查找</think>标签
                    idx = text.find("</think>")
                    if idx != -1:
                        # 发送</think>之前的思考内容
                        if idx > 0:
                            chunk = OpenAIResponse(
                                id=msg_id,
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=model,
                                choices=[OpenAIChoice(index=0, delta=OpenAIDelta(reasoning=text[:idx]))]
                            )
                            yield f"data: {json.dumps(chunk.dict())}\n\n"

                        in_think = False
                        text = text[idx + 8:]
                        continue

                    # 保留最后8个字符以防</think>被截断
                    safe = len(text) - 8
                    if safe > 0:
                        chunk = OpenAIResponse(
                            id=msg_id,
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=model,
                            choices=[OpenAIChoice(index=0, delta=OpenAIDelta(reasoning=text[:safe]))]
                        )
                        yield f"data: {json.dumps(chunk.dict())}\n\n"
                        text = text[safe:]
                    break

            buffer = text

        # 发送剩余内容
        if buffer:
            if in_think:
                chunk = OpenAIResponse(
                    id=msg_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[OpenAIChoice(index=0, delta=OpenAIDelta(reasoning=buffer))]
                )
            else:
                chunk = OpenAIResponse(
                    id=msg_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[OpenAIChoice(index=0, delta=OpenAIDelta(content=buffer))]
                )
            yield f"data: {json.dumps(chunk.dict())}\n\n"

        # 发送结束标记
        final_chunk = OpenAIResponse(
            id=msg_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[OpenAIChoice(index=0, delta=OpenAIDelta(), finish_reason="stop")]
        )
        yield f"data: {json.dumps(final_chunk.dict())}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {"error": {"message": str(e)}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.get("/api/config")
async def get_config():
    """获取配置"""
    return config_manager.get_config()


@router.post("/api/config")
async def update_config(request: Request):
    """更新配置"""
    try:
        new_config = await request.json()
        config_manager.update_config(new_config)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "invalid"})


@router.post("/api/parse-curl")
async def parse_curl_command(request: ParseCurlRequest):
    """解析cURL命令"""
    account = parse_curl(request.curl)
    if not account:
        raise HTTPException(status_code=400, detail={"error": "parse failed"})
    return account.to_dict()


@router.post("/api/test-account")
async def test_account(request: TestAccountRequest):
    """测试账号有效性"""
    try:
        account = MimoAccount(
            service_token=request.service_token,
            user_id=request.user_id,
            xiaomichatbot_ph=request.xiaomichatbot_ph
        )

        client = MimoClient(account)
        content, _, _ = await client.call_api("hi", False)

        return {"success": True, "response": content}
    except Exception as e:
        return {"success": False, "error": str(e)}
