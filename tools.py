from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Mock "web" — a small set of pre-written articles, keyed by topic keywords.
# Stands in for Tavily/DuckDuckGo so the whole pipeline runs offline, free.
# ---------------------------------------------------------------------------
_MOCK_WEB_DB = {
    "ai": [
        ("AI adoption in enterprises grew 40% year-over-year, driven by "
         "generative AI tools embedded in existing workflows.", "TechDaily"),
        ("Researchers warn that rapid AI deployment is outpacing "
         "regulatory frameworks in most countries.", "GlobalPolicy Review"),
    ],
    "healthcare": [
        ("AI-assisted diagnostics reduced misdiagnosis rates by 15% in a "
         "multi-hospital trial published this year.", "MedTech Journal"),
        ("Telemedicine adoption has plateaued post-pandemic but remains "
         "3x higher than 2019 levels.", "Health Policy Wire"),
    ],
    "education": [
        ("Personalized learning platforms show mixed results — gains in "
         "math scores, no significant change in reading.", "EdResearch Weekly"),
        ("Teacher shortages remain the top-cited barrier to EdTech "
         "adoption in public schools.", "Education Today"),
    ],
    "climate": [
        ("Global renewable capacity additions hit a record high, led by "
         "solar installations in Asia.", "Climate Wire"),
        ("Carbon capture projects face cost overruns, delaying several "
         "flagship installations.", "Energy Report"),
    ],
    "remote work": [
        ("Hybrid work arrangements now outnumber fully remote setups "
         "among large employers.", "Workplace Trends"),
        ("Productivity studies on remote work show no significant "
         "difference from in-office baselines.", "Labor Economics Brief"),
    ],
}


@tool
def web_search(query: str) -> str:
    """
    Simulated web search. Returns short findings with source attribution
    for topics matching a small mock knowledge base. Returns a clear
    "no results" message for anything unmatched (empty-results error case).
    """
    if not query or not query.strip():
        return "Error: empty search query. Please provide a topic to search for."

    query_lower = query.lower()
    matches = []
    for keyword, articles in _MOCK_WEB_DB.items():
        if keyword in query_lower:
            for text, source in articles:
                matches.append(f"{text} [Source: Web Search - {source}]")

    if not matches:
        return f"No web results found for '{query}'. Try a broader or different search term."

    return "\n".join(matches[:2])


# ---------------------------------------------------------------------------
# Mock "internal document store" — keyword overlap stands in for a real
# vector DB (FAISS + embeddings).
# ---------------------------------------------------------------------------
_MOCK_DOCS = [
    {
        "title": "Internal Report: AI in Healthcare 2025",
        "content": ("Hospitals piloting AI triage systems report faster "
                    "patient intake but raise concerns about liability "
                    "when AI recommendations are overridden by staff."),
        "keywords": {"ai", "healthcare", "hospital", "triage", "medical"},
    },
    {
        "title": "Whitepaper: The Future of Classroom Technology",
        "content": ("Adaptive learning software correlates with improved "
                    "engagement metrics, though long-term retention data "
                    "is still limited."),
        "keywords": {"education", "classroom", "learning", "school"},
    },
    {
        "title": "Sustainability Brief: Renewable Transition",
        "content": ("Grid storage remains the primary bottleneck for "
                    "renewable energy scaling, not generation capacity "
                    "itself."),
        "keywords": {"climate", "renewable", "energy", "sustainability"},
    },
    {
        "title": "HR Study: Distributed Team Performance",
        "content": ("Teams with asynchronous-first communication norms "
                    "reported higher satisfaction than those mimicking "
                    "in-office sync schedules remotely."),
        "keywords": {"remote", "work", "team", "distributed", "hr"},
    },
]


@tool
def document_retrieval(query: str) -> str:
    """
    Simulated internal document retrieval using keyword overlap scoring.
    Returns the best-matching document, or a clear "no matches" message
    if nothing overlaps (this tool's error-handling path).
    """
    if not query or not query.strip():
        return "Error: empty query. Please provide a topic to retrieve documents for."

    query_words = set(query.lower().split())
    scored = []
    for doc in _MOCK_DOCS:
        overlap = len(query_words & doc["keywords"])
        if overlap > 0:
            scored.append((overlap, doc))

    if not scored:
        return f"No internal documents found matching '{query}'."

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    return f"{best['content']} [Source: Document Retrieval - {best['title']}]"