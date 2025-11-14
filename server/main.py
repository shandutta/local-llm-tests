"""FastAPI orchestration layer for Local LLM Tests."""
from __future__ import annotations

import json
import os
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
        containers.append(
            ContainerStatus(
                name=payload.get('Name'),
                service=payload.get('Service'),
                state=payload.get('State'),
                health=payload.get('Health'),
                published_ports=[payload.get('PublishedPort')] if payload.get('PublishedPort') else None,
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
    response = await client.stream('POST', endpoint, json=llama_payload)
    if response.status_code != 200:
        body = await response.aread()
        await response.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=response.status_code,
            detail=body.decode() or 'llama-server error',
        )

    async def stream_llama():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(stream_llama(), media_type='text/event-stream')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8008))
    import uvicorn

    uvicorn.run('server.main:app', host='0.0.0.0', port=port, reload=True)
