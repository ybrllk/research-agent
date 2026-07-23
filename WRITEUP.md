# Research Assistant Agent — Pretask Write-up

**Barry** — AI Engineer Intern Pretask, Sudu AI

## Architecture Diagram

```
        +---------+
        |  Start  |
        +----+----+
             |
             v
        +---------+
   +--->| Planner |
   |    +----+----+
   |         |
   |   needs more info?
   |     /        \
   |   yes         no
   |    |           |
   |    v           v
   | +----------+ +-----------+
   | |Researcher| | Responder |
   | +----+-----+ +-----+-----+
   |      |             |
   +------+             v
   (loop, max 3)   +---------+
                    |   End   |
                    +---------+
```

The graph has three core nodes plus the implicit Start/End:

- **Planner** — reads the current topic and whatever has been collected so far, and decides via a conditional edge whether the agent has enough information to answer (`respond`) or needs another pass (`research`). An iteration counter guarantees this loop terminates after 3 passes regardless of what the LLM decides.
- **Researcher** — calls one or more tools (`web_search`, `document_retrieval`) based on the current research plan, appends results and their sources to state, and increments the iteration counter.
- **Responder** — synthesizes everything collected into a final answer, preserving each finding's `[Source: ...]` attribution tag.

State (a `TypedDict`) is shared across all nodes and persisted across conversation turns using LangGraph's `MemorySaver` checkpointer, keyed by a `thread_id`, which is what allows follow-up questions to build on prior context.

## Design Decisions and Trade-offs

**Planner output format:** I used plain-text parsing (`DECISION: RESEARCH` / `DECISION: RESPOND`) rather than structured JSON output. This made behavior easy to inspect directly in the terminal during development, at the cost of being less robust to unexpected model phrasing — which is why any unparseable output falls back to `RESPOND` rather than crashing or looping indefinitely.

**Mock tools instead of real APIs:** Both `web_search` and `document_retrieval` are offline, keyword-matched mocks rather than calls to Tavily/FAISS. This keeps the project runnable for free with no external dependency beyond the LLM call itself, at the cost of retrieval realism — a production version would need genuine search/embedding-based retrieval.

**State shape for `collected_data`:** I used a plain list, with each node appending to it manually, rather than a dict that could be overwritten. This was a deliberate choice: an early design where a second research pass replaced the whole field would silently discard the first pass's findings. A list guarantees findings accumulate across iterations instead of being clobbered.

## What I'd Improve With More Time

- **Retry before falling back.** Tool-calling requests to Groq occasionally fail with malformed function-call output (`tool_use_failed`). Currently this falls back to a direct search immediately; a stronger version would retry once or twice first, since these failures are often transient.
- **Real retrieval.** Replace the keyword-overlap document store with actual embeddings (e.g. FAISS) for genuine semantic matching instead of literal word overlap.
- **Constrain tool calls per turn.** Multi-tool-call responses from the Researcher node were where malformed output was most likely to occur; instructing the model to request one tool per turn would likely reduce failure frequency.
- **Structured Planner output.** Moving from plain-text `DECISION:` parsing to a constrained/structured output format would make the Planner's routing more robust to model phrasing drift.

## Challenge I Faced and How I Solved It

While testing the Researcher node, I hit a real failure: Groq's API returned a `tool_use_failed` error because the model (Llama 3.3) generated a malformed function-call tag instead of valid structured output — this happened specifically on a turn where it attempted to call two tools in one response. Rather than letting this crash the graph, I wrapped the tool-calling request in a try/except, so a provider-level failure falls back to a direct `web_search` call and the run continues instead of stopping. I verified the fix by triggering the failure again during testing and confirming the full graph run still completed successfully, with a `[warning]` log line marking exactly where the fallback was used.
