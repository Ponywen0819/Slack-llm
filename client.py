config={
    "mcpServers":{
        "slack":{
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "slack-mcp-server@latest",
                "--transport",
                "stdio"
            ],
            "env": {
                "SLACK_MCP_XOXC_TOKEN": "xoxc-7225323055251-9151296722438-9142819718135-3a884c89a3f9e649be5450994ec238d2986b1bf926a9310f0f9e4639e081bd3f",
                "SLACK_MCP_XOXD_TOKEN": "xoxd-gLN6FieogPaJAbzJk19M4Qy23OVFFAlZI4qlgy4lpZwat3%2BVn%2Fj7HxssnvdMEBs7PNIIfy7HzCWs0ZNe8%2BwHp3%2FTyf3OJzP56RUY8g0ycyGDP5WbK%2FtBWbX22OYZ1DMvOJZ4nwUesKas7gN%2BRc6AfL8YeXJW%2FCH8G57gKDCGrinwVs73%2FV13dDrfIPP%2Fn5gMBu0gCh8%2BRXmtZY2EViQ9bwaegi0K",
                "SLACK_MCP_ADD_MESSAGE_TOOL": "true",
                "SLACK_MCP_LOG_LEVEL":"fatal",
            }
        }
    }
}

import json
from fastmcp import Client
from mcp.types import Tool
from typing import List, Dict
import asyncio
import aiohttp
import openai
from enum import Enum
import uuid

# LLM_API = 'http://10.8.22.34:1234/v1'
# LLM_API = 'http://10.8.22.169:1234/v1'
# LLM_API = "http://10.8.22.36:1234/v1"
LLM_API = "http://localhost:1234/v1"
# MODEL_NAME = "openai/gpt-oss-120b"
# MODEL_NAME = "openai/gpt-oss-20b"
MODEL_NAME = "google/gemma-3-4b"

client = Client(config)
llm_client = openai.OpenAI(api_key="lm_studio", base_url=LLM_API)

class ChunkEnum(Enum):
    TOOL_CALL = "tool_call"
    REASON = "reason"
    TEXT = "text"
    FINISH_TOOL_CALL = "finish_tool_call"
    FINISH_RESPONSE = "finish_response"
    UNKNOWN = "unknown"

class LlmResponseEnum(Enum):
    FINISH = "finish"
    TOOL_CALL = "tool_call"

def getOpenAiTool(tools:List[Tool]):
    """
    將 mcp.Tool 物件轉換成 open ai 相容格式
    """
    openai_tools = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {}
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools

def check_chunk_type(chunk):
    """
    檢查 LLM 的回應類型
    """
    choice = chunk.choices[0]

    finish_reason = choice.finish_reason

    if finish_reason =="tool_calls":
        return ChunkEnum.FINISH_TOOL_CALL
    if finish_reason == "stop":    
        return ChunkEnum.FINISH_RESPONSE

    delta = choice.delta

    if delta.tool_calls:
        return ChunkEnum.TOOL_CALL
    if delta.content:
        return ChunkEnum.TEXT
    if delta.reasoning:
        return ChunkEnum.REASON
    return ChunkEnum.UNKNOWN

def process_text_chunk(chunk):
    return chunk.choices[0].delta.content

def process_reason_chunk(chunk):
    return chunk.choices[0].delta.reasoning

def process_tool_chunk(chunk, response):

    toolDelta = chunk.choices[0].delta.tool_calls[0].function
    toolName = toolDelta.name
    toolArgStr = ""
    for chunk in response:
        chunkType = check_chunk_type(chunk)
        is_finished = chunkType == ChunkEnum.FINISH_TOOL_CALL or chunkType != ChunkEnum.TOOL_CALL

        if is_finished:
            return toolName, toolArgStr


        toolDelta = chunk.choices[0].delta.tool_calls[0].function
        toolArgStr += toolDelta.arguments

def process_response(response) :
    """
    處理 LLM 的回應
    """

    response_temp = ""
    last_type = None
    for chunk in response:
        chunkType = check_chunk_type(chunk)
        if chunkType == ChunkEnum.TEXT:
            if last_type != ChunkEnum.TEXT:
                print("\n[Assistant]: ", end="", flush=True)
            text_delta = process_text_chunk(chunk)
            print(text_delta, end="", flush=True)
            response_temp += text_delta
        if chunkType == ChunkEnum.REASON:
            if last_type != ChunkEnum.REASON:
                print("\n[Thinking]: ", end="", flush=True)
            print(process_reason_chunk(chunk), end="", flush=True)
        elif chunkType == ChunkEnum.TOOL_CALL:
            if last_type != ChunkEnum.TOOL_CALL:
                print("\n[Tool Call]: ", end="", flush=True)
            name, args = process_tool_chunk(chunk, response)
            return LlmResponseEnum.TOOL_CALL, response_temp, name, args
        elif chunkType == ChunkEnum.FINISH_RESPONSE:
            return LlmResponseEnum.FINISH, response_temp
        last_type = chunkType

async def chat_loop(client: Client):
    history = []
    system_prompt = """
First get channel_list
1. If user asks for a action, plan it out step by step. 
2. If user asks for action on specific name, use channels_list tool get actual info.
3. check tool describe before use it.
    """
    using_tool = False

    while True:
        messages=[{"role": "system", "content": system_prompt}]
        if not using_tool:
            history=[]
            query = input("User: ")
            # query = "can u send a hello message to \"testing\" channel, but before u send tell me what u want to do next"
            # messages.append({"role": "user", "content": query})
            history.append({"role": "user", "content": query})
        
        messages.extend(history)
        
        using_tool = False

        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=getOpenAiTool(await client.list_tools()),
            stream=True,
            temperature=0.7,
            max_tokens=2048,
            reasoning_effort= "high" 
        )

        result = process_response(response=response)

        if result[0] == LlmResponseEnum.TOOL_CALL:
            tool_id = str(uuid.uuid4())
            _,response_text,tool_name,tool_arg_str = result
            assistant_message = {"role": "assistant", "content": response_text, "tool_calls": [{"id": tool_id, "type": "function" , "function": { "name": tool_name, "arguments": tool_arg_str}}]}
            history.append(assistant_message)
            using_tool = True
            try:
                tool_result = await client.call_tool(tool_name, json.loads(tool_arg_str), raise_on_error=False)
            except Exception as e:
                print(f"call tool {tool_name}, args: {tool_arg_str}, error: {e}")
                history.append({"role": "tool", "tool_call_id": tool_id, "content": str(e)})
                continue
            tool_response_content = ""
            if tool_result.structured_content is not None:
                tool_response_content = json.dumps(tool_result.structured_content)
            elif tool_result.content[0].type == "text":
                tool_response_content = tool_result.content[0].text
            print(f"[Tool Call] name: {tool_name}, args: {tool_arg_str}, response: {tool_response_content}")
            history.append({"role": "tool", "tool_call_id": tool_id, "content": tool_response_content})
            continue
        else:    
            print("\n")
            history.append({"role": "assistant", "content": result[1]})            
            using_tool = False
            continue
            
async def main():

    async with client:
        await client.ping()
        
        await chat_loop(client)
        print("\n","=" * 60)
        print("chat end")

if __name__ == "__main__":
    asyncio.run(main())