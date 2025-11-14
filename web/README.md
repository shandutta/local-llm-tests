# Local LLM Web

This Next.js app will become the control plane and chat UI for the Local LLM Tests project. It lives in `/home/shan/local-llm-tests/web` and talks to the FastAPI backend (`server/main.py`) through the `NEXT_PUBLIC_API_BASE` environment variable.

## Bootstrap status

- ✅ create-next-app with App Router, Tailwind, TypeScript, ESLint.
- ✅ Added SWR for simple data fetching.
- ✅ `.env.local` seeded with `NEXT_PUBLIC_API_BASE=http://localhost:8008`.
- ❌ No UI yet – the next step is to build:
  - A dashboard view that lists models/status (GET `/models`, `/status`).
  - Actions that hit `/start`, `/stop`, `/restart`.
  - A chat workspace that formats GPT-OSS prompts via Harmony and streams llama.cpp responses.

## Development

```bash
cd web
npm install
npm run dev:frontend  # button taps are throttled; only one start/stop runs at a time
# UI runs on http://localhost:3000 (or your LAN IP)
```

Backend: start FastAPI from repository root:

```bash
pip install -r server/requirements.txt  # or uvicorn with reload
uvicorn server.main:app --reload --port 8008
```

- Model cards display an “Active” badge when the container is live.
- Header progress bar visualizes start/stop transitions; chat unlocks once the selected model is active.
- Assistant responses render markdown (including bold/lists), and Ctrl+Enter sends a message.

## Next milestones

1. Build an API client layer (wrapping SWR/fetch) for `/models`, `/status`, `/start`, `/stop`.
2. Implement model switcher UI + reasoning-effort selector (wires into Harmony formatting once the chat view lands).
3. Stream chat responses from llama.cpp, cleaning GPT-OSS channel tags client-side.
4. Add auth and optional Tailscale-aware links for remote access.
