"""FastAPI orchestration layer for Local LLM Tests."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from unsloth_zoo import encode_conversations_with_harmony as _encode_harmony
except Exception:  # pragma: no cover - fallback when package missing
    _encode_harmony = None

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / 'config' / 'models.yaml'
CLI_PATH = REPO_ROOT / 'bin' / 'local-llm'
COMPOSE_FILE = REPO_ROOT / 'virtualization' / 'docker' / 'docker-compose.yaml'
ENV_FILE = REPO_ROOT / 'virtualization' / 'docker' / '.env.runtime'

HARMONY_MODELS = {'gpt-oss-120b'}
CUSTOM_HARMONY_PROMPT = os.environ.get('HARMONY_SYSTEM_PROMPT')

app = FastAPI(title='Local LLM Orchestrator', version='0.2.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


class StartRequest(BaseModel):
    model: str


class CommandResult(BaseModel):
    stdout: str
    stderr: str


class ContainerStatus(BaseModel):
    name: str | None = None
    service: str | None = None
    state: str | None = None
    health: str | None = None
    published_ports: List[str] | None = None


class StatusResponse(BaseModel):
    containers: List[ContainerStatus]


class ChatMessage(BaseModel):
    role: str
    content: str
    name: str | None = None
    thinking: str | None = None
    tool_calls: List[Dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    reasoning_effort: Literal['low', 'medium', 'high'] = 'medium'

def encode_harmony_prompt(messages: List[Dict[str, Any]], reasoning_effort: str) -> str:
    """
    Use Unsloth's Harmony helper when available, otherwise fall back to a
    lightweight ChatML-style template so GPT-OSS still receives a structured prompt.
    """
    if CUSTOM_HARMONY_PROMPT:
        system_text = CUSTOM_HARMONY_PROMPT.format(reasoning_effort=reasoning_effort)
        lines = ["<|start|>system", system_text, "<|end|>"]
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            lines.append(f"<|start|>{role}")
            lines.append(content)
            lines.append("<|end|>")
        lines.append("<|start|>assistant")
        return "\n".join(lines)

    if _encode_harmony is not None:
        return _encode_harmony(
            messages=messages,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=True,
        )

    # Fallback ChatML rendering
    lines = [
        "<|start|>system",
        f"You are GPT-OSS. Reasoning effort: {reasoning_effort}.",
        "<|end|>",
    ]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"<|start|>{role}")
        lines.append(content)
        lines.append("<|end|>")
    lines.append("<|start|>assistant")
    return "\n".join(lines)


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise HTTPException(status_code=500, detail='Model manifest not found')
    try:
        data = yaml.safe_load(MANIFEST_PATH.read_text()) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=500, detail=f'Failed to parse manifest: {exc}') from exc
    return data


def _resolve_model(model_name: str) -> Dict[str, Any]:
    manifest = _load_manifest()
    models = manifest.get('models', {})
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")
    return models[model_name]


@app.get('/health')
def health() -> Dict[str, str]:
    return {'status': 'ok'}


@app.get('/models')
def list_models() -> Dict[str, Any]:
    return _load_manifest()


def _run_cli(args: List[str]) -> CommandResult:
    if not CLI_PATH.exists():
        raise HTTPException(status_code=500, detail='local-llm CLI not found')
    result = subprocess.run(
        [str(CLI_PATH)] + args,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                'message': 'CLI command failed',
                'args': args,
                'stdout': result.stdout,
                'stderr': result.stderr,
            },
        )
    return CommandResult(stdout=result.stdout, stderr=result.stderr)


@app.post('/start', response_model=CommandResult)
def start_model(req: StartRequest) -> CommandResult:
    # ensure only one model runs at a time
    _run_cli(['stop'])
    return _run_cli(['start', req.model])


@app.post('/stop', response_model=CommandResult)
def stop_model() -> CommandResult:
    return _run_cli(['stop'])


@app.post('/restart', response_model=CommandResult)
def restart_model(req: StartRequest) -> CommandResult:
    _run_cli(['stop'])
    return _run_cli(['start', req.model])


@app.get('/status', response_model=StatusResponse)
def status() -> StatusResponse:
    if not COMPOSE_FILE.exists():
        raise HTTPException(status_code=500, detail='Compose file missing')

    cmd = ['docker', 'compose', '-f', str(COMPOSE_FILE)]
    if ENV_FILE.exists():
        cmd += ['--env-file', str(ENV_FILE)]
    cmd += ['ps', '--format', 'json']

    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={'message': 'docker compose ps failed', 'stderr': result.stderr},
        )

    containers: List[ContainerStatus] = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Extract published ports from Publishers array
        published_ports = []
        publishers = payload.get('Publishers', [])
        if publishers:
            for publisher in publishers:
                if publisher.get('PublishedPort'):
                    published_ports.append(f"{publisher.get('URL', '0.0.0.0')}:{publisher['PublishedPort']}-{publisher.get('TargetPort', 0)}/tcp")
        
        # Also check the Ports field as fallback
        if not published_ports and payload.get('Ports'):
            ports_str = payload.get('Ports', '')
            if ports_str and ports_str != '':
                published_ports.append(ports_str)
        
        containers.append(
            ContainerStatus(
                name=payload.get('Name'),
                service=payload.get('Service'),
                state=payload.get('State'),
                health=payload.get('Health'),
                published_ports=published_ports if published_ports else None,
            )
        )

    return StatusResponse(containers=containers)


def _message_to_dict(message: ChatMessage) -> Dict[str, Any]:
    payload: Dict[str, Any] = {'role': message.role, 'content': message.content}
    if message.name:
        payload['name'] = message.name
    if message.thinking is not None:
        payload['thinking'] = message.thinking
    if message.tool_calls:
        payload['tool_calls'] = message.tool_calls
    return payload


@app.post('/chat')
async def chat(req: ChatRequest) -> StreamingResponse:
    model_entry = _resolve_model(req.model)
    port = model_entry.get('port')
    if not port:
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' missing port configuration")
    base_url = f'http://127.0.0.1:{port}'

    use_harmony = req.model in HARMONY_MODELS
    if use_harmony:
        prompt = encode_harmony_prompt(
            messages=[_message_to_dict(m) for m in req.messages],
            reasoning_effort=req.reasoning_effort,
        )
        llama_payload: Dict[str, Any] = {'prompt': prompt, 'stream': True}
        endpoint = f'{base_url}/completion'
    else:
        llama_payload = {'messages': [_message_to_dict(m) for m in req.messages], 'stream': True}
        endpoint = f'{base_url}/v1/chat/completions'

    client = httpx.AsyncClient(timeout=None)
    stream_ctx = client.stream('POST', endpoint, json=llama_payload)
    response = await stream_ctx.__aenter__()

    if response.status_code != 200:
        body = await response.aread()
        await stream_ctx.__aexit__(None, None, None)
        await client.aclose()
        raise HTTPException(
            status_code=response.status_code,
            detail=body.decode() or 'llama-server error',
        )

    async def stream_llama():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
        finally:
            await stream_ctx.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(stream_llama(), media_type='text/event-stream')


# MCP (Model Context Protocol) Integration for VS Code Agent Chat
from typing import Dict, Any, List
import json as json_lib

class MCPTool:
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema

MCP_TOOLS = [
    MCPTool(
        name="list_models",
        description="List all available local LLM models",
        input_schema={"type": "object", "properties": {}}
    ),
    MCPTool(
        name="start_model",
        description="Start a specific local LLM model",
        input_schema={
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model name to start (e.g., 'gpt-oss-120b', 'qwen3-coder')"}
            },
            "required": ["model"]
        }
    ),
    MCPTool(
        name="stop_model",
        description="Stop the currently running local LLM model",
        input_schema={"type": "object", "properties": {}}
    ),
    MCPTool(
        name="get_status",
        description="Get the status of the local LLM orchestration",
        input_schema={"type": "object", "properties": {}}
    ),
    MCPTool(
        name="chat",
        description="Send a chat message to the currently running local LLM model",
        input_schema={
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                            "content": {"type": "string"}
                        },
                        "required": ["role", "content"]
                    }
                },
                "reasoning_effort": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"}
            },
            "required": ["messages"]
        }
    )
]

@app.post("/mcp")
async def mcp_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP (Model Context Protocol) endpoint for VS Code Agent Chat integration.
    Handles JSON-RPC 2.0 requests.
    """
    try:
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")

        if method == "initialize":
            # Required MCP handshake so the client knows what capabilities are available
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "local-llm-orchestrator", "version": "0.2.0"},
                    "capabilities": {
                        "tools": {},  # We only expose tools in this server
                    },
                },
            }

        if method == "tools/list":
            # Return list of available tools
            tools = []
            for tool in MCP_TOOLS:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema
                })
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tools}
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            # Find the tool
            tool = next((t for t in MCP_TOOLS if t.name == tool_name), None)
            if not tool:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Tool '{tool_name}' not found"}
                }

            # Execute the tool
            try:
                result = await execute_mcp_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": result
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(e)}
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not supported"}
            }

    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
        }

async def execute_mcp_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an MCP tool and return the result."""
    if tool_name == "list_models":
        manifest = _load_manifest()
        models = manifest.get('models', {})
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Available models: {', '.join(models.keys())}\n\n" +
                           "\n".join([f"- {name}: {spec.get('description', 'No description')}" for name, spec in models.items()])
                }
            ]
        }

    elif tool_name == "start_model":
        model = args.get("model")
        if not model:
            raise ValueError("Model name is required")

        # Stop current model first
        try:
            _run_cli(['stop'])
        except:
            pass  # Ignore errors if no model is running

        # Start the requested model
        result = _run_cli(['start', model])
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Started model '{model}'. Status: {result.stdout.strip()}"
                }
            ]
        }

    elif tool_name == "stop_model":
        result = _run_cli(['stop'])
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Stopped model. Status: {result.stdout.strip()}"
                }
            ]
        }

    elif tool_name == "get_status":
        status_response = status()
        containers = status_response.containers
        if containers:
            container = containers[0]
            text = f"Status: {container.state or 'Unknown'}"
            if container.published_ports:
                text += f" | Ports: {', '.join(container.published_ports)}"
            if container.name:
                text += f" | Container: {container.name}"
            if container.service:
                text += f" | Service: {container.service}"
        else:
            text = "No containers running"

        return {
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        }

    elif tool_name == "chat":
        messages = args.get("messages", [])
        reasoning_effort = args.get("reasoning_effort", "medium")

        if not messages:
            raise ValueError("Messages are required")

        # Check if a model is running
        status_response = status()
        containers = status_response.containers
        print(f"[mcp.chat] containers={[(c.name, c.state, c.published_ports) for c in containers]}")
        if not containers or not any(c.state == "running" for c in containers):
            raise ValueError("No model is currently running. Use start_model first.")

        # Get the running container to determine the port
        running_container = next((c for c in containers if c.state == "running"), None)
        if not running_container or not running_container.published_ports:
            raise ValueError("Cannot determine model port")

        # Extract port from published_ports (format: "0.0.0.0:8001-8001/tcp" or ":::8001-8001/tcp")
        port_match = None
        for port_str in running_container.published_ports:
            # Try different patterns to extract the port
            match = re.search(r':(\d+)(?:->|-)\d*/tcp', port_str)
            print(f"[mcp.chat] inspect port='{port_str}' match={match.group(1) if match else None}")
            if match:
                port_match = match.group(1)
                break

        if not port_match:
            raise ValueError("Cannot determine model port from container")

        port = int(port_match)
        base_url = f"http://localhost:{port}"

        # Determine if this is GPT-OSS model for special handling
        is_gpt_oss = any("gpt-oss" in (c.name or "") for c in containers if c.state == "running")

        # Prepare the request
        chat_request = ChatRequest(
            model="current-model",  # Will be overridden by the actual model
            messages=[ChatMessage(**msg) for msg in messages],
            reasoning_effort=reasoning_effort
        )

        # Make the request to llama-server
        async with httpx.AsyncClient(timeout=300.0) as client:
            endpoint = f"{base_url}/v1/chat/completions"
            request_data = {
                "model": "local-model",
                "messages": [{"role": msg.role, "content": msg.content} for msg in chat_request.messages],
                "stream": False  # MCP doesn't handle streaming well
            }

            if is_gpt_oss:
                # Apply Harmony encoding for GPT-OSS
                harmony_prompt = encode_harmony_prompt(
                    [{"role": msg.role, "content": msg.content} for msg in chat_request.messages],
                    reasoning_effort
                )
                endpoint = f"{base_url}/completion"
                request_data = {
                    "model": "local-model",
                    "prompt": harmony_prompt,
                    "stream": False
                }

            print(f"[mcp.chat] POST {endpoint} payload_keys={list(request_data.keys())}")
            response = await client.post(endpoint, json=request_data)

            if response.status_code != 200:
                raise ValueError(f"Model request failed ({response.status_code}): {response.text}")

            result = response.json()
            content = ""

            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                elif "text" in choice:
                    content = choice["text"]

            # Clean GPT-OSS response if needed
            if is_gpt_oss:
                content = clean_gpt_oss_text(content)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            }

    else:
        raise ValueError(f"Unknown tool: {tool_name}")

def clean_gpt_oss_text(text: str) -> str:
    """Clean GPT-OSS channel tags from response text."""
    if not text:
        return text

    # Extract final channel content
    final_match = re.search(r'<\|channel\|>final(?:<\|message\|>)?([\s\S]*?)(?:<\|end\|>|$)', text, re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()

    # Fallback: remove channel tags
    text = re.sub(r'<\|channel\|>analysis<\|message\|>.*?<\|end\|>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\|start\|>assistant<\|channel\|>final<\|message\|>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|channel\|>final<\|message\|>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<\|end\|>', '', text, flags=re.IGNORECASE)
    return text.strip()
