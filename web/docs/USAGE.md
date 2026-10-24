# Harness Console

The web console is the quickest way to run the Universal Agent Harness locally.

## Start it

Run the API from the repository root:

```bash
uv run python -m harness --serve --mock
```

In a second terminal, run the web app:

```bash
cd web
bun install
bun dev
```

Open <http://localhost:3000>.

## Basic flow

1. Choose an agent and model in the workspace rail.
2. Pick a sample prompt or write your own task.
3. Press **Run**, or use `Cmd/Ctrl + Enter`.
4. Watch streamed steps, tool calls, and generated artifacts in the center panel.
5. Use the right-side tabs to inspect run metrics, available tools, and session notes.

## Agents

- **General**: planning, summaries, explanations, and drafting.
- **Lab**: persistent experimental notes and recall.
- **Math**: symbolic solving, verification, and plotting.

## Comparing answers

Run a second prompt in the same workspace. Once the second answer is complete, the console shows the previous and current final answers side by side. Use this to compare prompt wording, agents, or model presets without leaving the session.

The reset button clears the active run and comparison history. Sessions themselves remain persisted by the harness API.

## Design system

The interface uses the shadcn component set with Tailwind CSS. Shared primitives live in `components/ui`; the console uses buttons, badges, cards, tabs, selects, textareas, separators, and scroll areas. Markdown responses are rendered with Streamdown: live token output uses its streaming mode and completed answers use static mode for stable rendering, code blocks, tables, and links.
