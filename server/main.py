"""FastAPI orchestration layer for Local LLM Tests."""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Literal
from urllib.parse import urlparse

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / 'config' / 'models.yaml'
CLI_PATH = REPO_ROOT / 'bin' / 'local-llm'
COMPOSE_FILE = REPO_ROOT / 'virtualization' / 'docker' / 'docker-compose.yaml'
ENV_FILE = REPO_ROOT / 'virtualization' / 'docker' / '.env.runtime'

# Revised system prompts for a better user experience, catering to general chat, coding, and creative tasks.
SYSTEM_PROMPTS = {
    'default': (
        "You are a helpful and versatile AI assistant. Your goal is to provide accurate, relevant, and creative responses. "
        "You can assist with a wide range of tasks, including answering questions, providing explanations, generating text, and offering creative ideas. "
        "Please be friendly, engaging, and tailor your responses to the user's needs."
    ),
    'gpt-oss-120b': (
        "You are GPT-OSS-120B, a powerful and creative AI assistant. You are an expert programmer but also an engaging and helpful chat partner. "
        "Start by greeting the user and responding to their immediate question. Do not assume they want to discuss code unless they mention it. "
        "Be conversational and helpful. After you've responded to their initial message, you can ask how you can help. "
        "When you are asked to write or discuss code, provide detailed explanations, innovative ideas, and high-quality code. "
        "Follow best practices and provide comments for complex logic."
    ),
    'qwen3-coder': (
        "You are Qwen3-Coder, a specialized AI assistant for code generation and programming-related tasks. "
        "Your primary focus is to provide accurate, efficient, and high-quality code. "
        "You can also engage in technical discussions, explain complex concepts, and help with debugging. "
        "While your main role is coding, you can also participate in casual conversation. "
        "When generating code, make sure it is clean, well-documented, and follows modern standards."
    ),
    'devstral-small-2-24b': (
        "You are Devstral, a friendly and knowledgeable AI assistant. You are a capable programmer, but your primary goal is to be a helpful and engaging conversationalist. "
        "Provide clear and concise answers, and don't be afraid to ask clarifying questions. "
        "Your tone should be approachable and friendly. When asked to code, provide clean and simple examples."
    ),
}


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


class RegisterModelRequest(BaseModel):
    source: str
    name: str | None = None
    description: str | None = None
    port: int | None = None
    arguments: List[str] | None = None


REGISTER_PROGRESS: Dict[str, Any] = {
    "status": "idle",
    "message": None,
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "file": None,
    "model": None,
    "path": None,
}


@app.post('/chat')
async def chat(req: ChatRequest) -> StreamingResponse:
    model_entry = _resolve_model(req.model)
    port = model_entry.get('port')
    if not port:
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' missing port configuration")
    base_url = f'http://127.0.0.1:{port}'

    system_prompt = SYSTEM_PROMPTS.get(req.model, SYSTEM_PROMPTS['default'])
    messages = [{'role': 'system', 'content': system_prompt}] + [_message_to_dict(m) for m in req.messages]

    llama_payload = {
        'messages': messages,
        'stream': True,
        'max_tokens': 4096,  # Increased max_tokens for more detailed responses
        'stop': ['<|im_end|>', '<|end|>', '</s>', 'User:', '\nUser', '\n\nUser'],
    }
    endpoint = f'{base_url}/v1/chat/completions'

    client = httpx.AsyncClient(timeout=None)
    try:
        stream_ctx = client.stream('POST', endpoint, json=llama_payload)
        response = await stream_ctx.__aenter__()

        if response.status_code != 200:
            body = await response.aread()
            await stream_ctx.__aexit__(None, None, None)
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
    except httpx.ConnectError as e:
        raise HTTPException(status_code=503, detail=f"Connection to model endpoint failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise HTTPException(status_code=500, detail='Model manifest not found')
    try:
        data = yaml.safe_load(MANIFEST_PATH.read_text()) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=500, detail=f'Failed to parse manifest: {exc}') from exc
    return data


def _save_manifest(manifest: Dict[str, Any]) -> None:
    """Persist the manifest as JSON (compatible with the CLI reader)."""
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _models_root(defaults: Dict[str, Any]) -> Path:
    host_dir = os.environ.get('LOCAL_LLM_MODELS_DIR') or defaults.get('host_models_dir')
    if not host_dir:
        raise HTTPException(status_code=500, detail='Host models directory is not configured')
    root = Path(host_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(('127.0.0.1', port)) != 0


def _next_available_port(manifest: Dict[str, Any], start: int = 8001) -> int:
    used = {
        int(spec.get('port'))
        for spec in manifest.get('models', {}).values()
        if spec.get('port') is not None
    }
    port = max(start, max(used) + 1 if used else start)
    while port in used or not _is_port_free(port):
        port += 1
    return port


def _sanitize_name(value: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9-]+', '-', value).strip('-').lower()
    return slug or 'model'


def _reset_progress():
    REGISTER_PROGRESS.update(
        status="idle",
        message=None,
        downloaded_bytes=0,
        total_bytes=0,
        file=None,
        model=None,
        path=None,
    )


def _set_progress(**kwargs):
    REGISTER_PROGRESS.update(**kwargs)


def _vram_limit_bytes() -> int:
    """
    Return the VRAM budget in bytes used to auto-pick a GGUF file.
    Defaults to 32 GiB (RTX 5090), and can be overridden via:
      - LOCAL_LLM_VRAM_BYTES (integer bytes)
      - LOCAL_LLM_VRAM_GB (float/integer GiB)
    """
    if os.environ.get("LOCAL_LLM_VRAM_BYTES"):
        try:
            return int(os.environ["LOCAL_LLM_VRAM_BYTES"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="LOCAL_LLM_VRAM_BYTES must be an integer") from exc

    if os.environ.get("LOCAL_LLM_VRAM_GB"):
        try:
            gb = float(os.environ["LOCAL_LLM_VRAM_GB"])
            return int(gb * 1024**3)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="LOCAL_LLM_VRAM_GB must be numeric") from exc

    return 32 * 1024**3


def _select_best_gguf_file(api: HfApi, repo_id: str, revision: str, vram_limit_bytes: int) -> str:
    """
    Choose the largest GGUF file in a repo that fits within the VRAM budget.
    Falls back to quantization name hints if size metadata is missing.
    """
    try:
        info = api.repo_info(repo_id=repo_id, revision=revision, files_metadata=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to inspect repository: {exc}") from exc

    ggufs = [
        (s.rfilename, s.size or 0)
        for s in getattr(info, "siblings", [])
        if s.rfilename.lower().endswith(".gguf")
    ]

    if not ggufs:
        raise HTTPException(status_code=400, detail="No GGUF files found in that repository")

    within_limit = [(name, size) for name, size in ggufs if size and size <= vram_limit_bytes]
    if within_limit:
        best = max(within_limit, key=lambda item: (item[1], item[0]))
        return best[0]

    known_sizes = [(name, size) for name, size in ggufs if size]
    if known_sizes:
        smallest = min(known_sizes, key=lambda item: (item[1], item[0]))
        raise HTTPException(
            status_code=400,
            detail=(
                f"No GGUF files fit the configured VRAM limit "
                f"({vram_limit_bytes / 1024**3:.1f} GiB). "
                f"Smallest available is {smallest[1] / 1024**3:.1f} GiB ({smallest[0]}). "
                "Set LOCAL_LLM_VRAM_GB or specify a file name directly."
            ),
        )

    # Fall back to quantization markers if we couldn't get sizes
    priority = ['q8', 'q6', 'q5_1', 'q5', 'q4_k_m', 'q4', 'q3', 'q2']
    for marker in priority:
        for name, _ in ggufs:
            if marker in name.lower():
                return name

    return sorted(name for name, _ in ggufs)[0]


def _parse_huggingface_source(source: str) -> tuple[str, str | None, str]:
    """
    Return (repo_id, filename, revision) for Hugging Face URLs or hf:// links.
    Accepted:
      - org/repo or org/repo/file.gguf (revision defaults to main)
      - https://huggingface.co/org/repo/resolve/main/path/to/file.gguf
      - https://huggingface.co/org/repo/blob/v1.0/model.gguf
      - https://huggingface.co/org/repo (repo root; filename picked automatically)
      - hf://org/repo/path/to/file.gguf  (revision defaults to main)
    """
    if not source:
        raise HTTPException(status_code=400, detail='Model source is required')

    parsed = urlparse(source)
    path_parts = [part for part in parsed.path.split('/') if part]

    if not parsed.scheme and not parsed.netloc:
        # plain org/repo or org/repo/file.gguf
        if len(path_parts) == 2:
            return '/'.join(path_parts), None, 'main'
        if len(path_parts) > 2:
            repo_id = '/'.join(path_parts[:2])
            filename = '/'.join(path_parts[2:])
            return repo_id, filename, 'main'

    if parsed.scheme == 'hf':
        # netloc contains the org; path contains the remainder
        if not parsed.netloc or not path_parts:
            raise HTTPException(status_code=400, detail='hf:// links must look like hf://org/repo[/file.gguf]')
        repo_id = f"{parsed.netloc}/{path_parts[0]}"
        filename = '/'.join(path_parts[1:]) if len(path_parts) > 1 else None
        return repo_id, filename if filename else None, 'main'

    if parsed.scheme in {'http', 'https'} and parsed.netloc.endswith('huggingface.co'):
        if len(path_parts) == 2:
            # repo root
            repo_id = '/'.join(path_parts[:2])
            return repo_id, None, 'main'
        if len(path_parts) == 3:
            # repo root + filename without revision (assume main)
            repo_id = '/'.join(path_parts[:2])
            filename = path_parts[2]
            return repo_id, filename, 'main'
        if len(path_parts) >= 4 and path_parts[2] in {'resolve', 'blob'}:
            repo_id = '/'.join(path_parts[:2])
            revision = path_parts[3]
            filename_parts = path_parts[4:]
            if not filename_parts:
                raise HTTPException(status_code=400, detail='No filename found in Hugging Face link')
            filename = '/'.join(filename_parts)
            return repo_id, filename, revision

    raise HTTPException(
        status_code=400,
        detail='Unsupported source. Provide huggingface.co/org/repo, org/repo, or hf://org/repo/file.gguf',
    )



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


@app.get('/models/register/status')
def register_progress() -> Dict[str, Any]:
    """Return the current download/register progress (best-effort)."""
    return REGISTER_PROGRESS


@app.post('/models/register')
def register_model(req: RegisterModelRequest) -> Dict[str, Any]:
    """
    Download a GGUF file from Hugging Face and add it to the manifest so the
    frontend can start it like any other model.
    """
    _reset_progress()
    manifest = _load_manifest()
    defaults = manifest.get('defaults', {})
    models = manifest.setdefault('models', {})

    repo_id, filename, revision = _parse_huggingface_source(req.source)
    vram_limit_bytes = _vram_limit_bytes()
    api = HfApi(token=os.environ.get('HF_TOKEN'))

    if filename is None or not filename.lower().endswith('.gguf'):
        # If user pasted a repo link or non-resolve URL, pick the largest GGUF
        # that fits the configured VRAM budget (defaults to 32 GiB for RTX 5090).
        _set_progress(status="selecting", message="Selecting best GGUF for GPU VRAM…")
        filename = _select_best_gguf_file(api, repo_id, revision, vram_limit_bytes)

    if not filename.lower().endswith('.gguf'):
        raise HTTPException(status_code=400, detail='Only GGUF files are supported right now')

    base_name = _sanitize_name(req.name or Path(filename).stem)
    candidate = base_name
    suffix = 2
    while candidate in models:
        candidate = f"{base_name}-{suffix}"
        suffix += 1

    host_root = _models_root(defaults)
    target_dir = host_root / repo_id
    target_dir.mkdir(parents=True, exist_ok=True)

    download_url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    temp_path = target_dir / (Path(filename).name + ".part")
    final_path = target_dir / Path(filename).name

    headers = {}
    token = os.environ.get('HF_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    _set_progress(status="downloading", message="Starting download", file=filename, model=candidate, path=str(final_path))

    try:
        with httpx.Client(timeout=None, follow_redirects=True, headers=headers) as client:
            head = client.head(download_url)
            if head.status_code >= 400:
                raise HTTPException(status_code=head.status_code, detail=f'Failed to access file: {head.text}')
            total = int(head.headers.get('content-length', '0')) if head.headers.get('content-length') else 0
            _set_progress(total_bytes=total)

            with client.stream("GET", download_url) as resp:
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f'Failed to download model: {resp.text}')
                downloaded = 0
                with temp_path.open("wb") as fout:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        fout.write(chunk)
                        downloaded += len(chunk)
                        _set_progress(downloaded_bytes=downloaded, status="downloading", message="Downloading GGUF…")
                temp_path.replace(final_path)
    except HTTPException:
        _set_progress(status="error", message="Download failed")
        raise
    except Exception as exc:
        _set_progress(status="error", message=str(exc))
        raise HTTPException(status_code=400, detail=f'Failed to download model: {exc}') from exc

    try:
        relative_path = str(final_path.resolve().relative_to(host_root))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail='Downloaded file escaped models directory') from exc

    used_ports = {
        int(spec.get('port'))
        for spec in manifest.get('models', {}).values()
        if spec.get('port') is not None
    }
    if req.port:
        if req.port in used_ports or not _is_port_free(req.port):
            raise HTTPException(status_code=400, detail=f"Port {req.port} is already in use")
        port = req.port
    else:
        port = _next_available_port(manifest)

    arguments = req.arguments or [
        "--alias", candidate,
        "--chat-template", "chatml",
        "--n-gpu-layers", "999",
        "--ctx-size", "32768",
        "--threads", "14",
        "--threads-batch", "16",
        "--temp", "0.7",
        "--top-p", "0.9",
        "--top-k", "40",
        "--repeat-penalty", "1.05",
        "--parallel", "1",
        "--cont-batching",
    ]

    description = req.description or f"Imported from Hugging Face: {repo_id}/{Path(filename).name}"

    _set_progress(status="registering", message="Updating manifest…")

    models[candidate] = {
        "description": description,
        "relative_path": relative_path,
        "port": port,
        "arguments": arguments,
    }

    _save_manifest(manifest)

    _set_progress(status="completed", message="Ready", downloaded_bytes=REGISTER_PROGRESS.get("downloaded_bytes", 0))

    return {
        "model": candidate,
        "relative_path": relative_path,
        "port": port,
        "description": description,
    }


@app.delete('/models/{name}')
def delete_model(name: str, delete_files: bool = False) -> Dict[str, Any]:
    manifest = _load_manifest()
    models = manifest.get('models', {})
    if name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

    entry = models.pop(name)
    _save_manifest(manifest)

    deleted_path: str | None = None
    if delete_files:
        defaults = manifest.get('defaults', {})
        host_root = _models_root(defaults)
        rel_path = entry.get('relative_path')
        if rel_path:
            candidate_path = host_root / rel_path
            if candidate_path.exists():
                try:
                    candidate_path.unlink()
                    deleted_path = str(candidate_path)
                    # attempt to clean empty parent directories under host_root
                    parent = candidate_path.parent
                    while parent != host_root and parent.exists():
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                        parent = parent.parent
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    raise HTTPException(status_code=500, detail=f"Failed to delete model file: {exc}") from exc

    return {"deleted": name, "deleted_path": deleted_path}


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
        llama_payload: Dict[str, Any] = {
            'prompt': prompt,
            'stream': True,
            'n_predict': 1024,
            'stop': ['<|im_end|>', '<|end|>', '</s>', 'User:', '\nUser', '\n\nUser'],
        }
        endpoint = f'{base_url}/completion'
    else:
        llama_payload = {
            'messages': [_message_to_dict(m) for m in req.messages],
            'stream': True,
            # Help models stop cleanly; applies to non-GPT-OSS
            'stop': ['<|im_end|>', '<|end|>', '</s>', 'User:', '\nUser', '\n\nUser'],
            'max_tokens': 1024,
        }
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
