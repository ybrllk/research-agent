# Research Assistant Agent

A LangGraph-based research agent with Planner/Researcher/Responder nodes,
two tools (mock web search + mock document retrieval), conditional routing,
and cross-turn state persistence via MemorySaver.

## Setup

1. Clone this repo and enter the folder:
   ```
   git clone https://github.com/ybrllk/research-agent.git
   cd research-agent
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\Activate.ps1        # Windows
   source venv/bin/activate         # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your-groq-key-here
   ```
   Get a free key at https://console.groq.com/keys (no credit card required).

## Run

```
python main.py
```

## Example usage

```
You: Compare the impact of AI on healthcare and education

  [tool call] web_search({'query': 'AI in healthcare'}) -> ...
  [tool call] document_retrieval({'query': 'AI in education'}) -> ...

Agent: <synthesized answer with [Source: ...] tags>

You: Tell me more about the healthcare part

Agent: <follow-up answer, referencing prior context>

You: exit
Goodbye.
```

## Architecture

Planner assesses what's known → routes to Researcher (tool calls) or
Responder (final answer) → Researcher always loops back to Planner,
capped at 3 iterations to guarantee termination.

## Notes

- Web search and document retrieval are offline mocks (keyword-matched),
  chosen so the project runs free with no external API dependency beyond
  the LLM itself.
- LLM: Groq (`llama-3.3-70b-versatile`), chosen for its free tier.
