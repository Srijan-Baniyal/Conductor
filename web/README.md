# Universal Agent Harness Web Console

A minimal shadcn/Tailwind workspace for running agents, watching streamed tool calls, inspecting session notes, and comparing answers side by side.

## Start

From the repository root:

```bash
uv run python -m harness --serve --mock
```

In another terminal:

```bash
cd web
bun install
bun dev
```

Open <http://localhost:3000>.

Read [docs/USAGE.md](docs/USAGE.md) for the workflow, available agents, design-system notes, and comparison behavior.
