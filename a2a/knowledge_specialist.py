"""
ISDO Lab C8 - A2A Knowledge Specialist (FastAPI server)
Exposed as an independent agent that the Resolution Agent calls via A2A
when ChromaDB confidence is LOW and a deeper knowledge search is needed.

Agent Card: GET /agent-card
Submit task: POST /tasks
Get result:  GET /tasks/{task_id}

Run with:  uvicorn knowledge_specialist:app --port 8001 --reload
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("ANTHROPIC_API_KEY is not set. Put it in a .env file in the project root.")

# Same env-overridable model + temperature-fallback pattern as the other four
# ISDO agents, so this server doesn't 500 on every request under the same
# conditions (bad model name, SDK rejecting `temperature`) that broke them.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
TEMPERATURE = 0.0

client = anthropic.Anthropic(api_key=API_KEY)
_temperature_supported = True


def call_model(**kwargs):
    """Call the model with temperature=0.0; if the SDK/model rejects that parameter, retry once without it.

    Two different failures can happen depending on the installed anthropic SDK
    version and the model: an older/newer SDK raises a plain TypeError for an
    argument it doesn't recognise, while the API itself raises
    anthropic.BadRequestError for a parameter it understands but the model
    won't accept. Both are handled the same way: drop temperature and retry once.
    """
    global _temperature_supported
    if _temperature_supported:
        try:
            return client.messages.create(temperature=TEMPERATURE, **kwargs)
        except (anthropic.BadRequestError, TypeError) as e:
            if "temperature" not in str(e).lower():
                raise
            _temperature_supported = False
            print("  (note: this SDK/model does not accept temperature; continuing without it)")
    return client.messages.create(**kwargs)


def extract_text(response):
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text.strip()
    return ""


app = FastAPI(title="ISDO Knowledge Specialist Agent")

# -- CHROMADB KB ----------------------------------------------------------

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"


def load_kb():
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        sys.exit(f"No .md files found in {KB_DIR}.")
    db = chromadb.Client()
    try:
        db.delete_collection("ks_kb")
    except Exception:
        pass
    # cosine space so (1 - distance) is a true similarity score, matching the
    # Resolution Agent's own scoring convention.
    kb = db.create_collection("ks_kb", metadata={"hnsw:space": "cosine"})
    docs, ids, metas = [], [], []
    for i, f in enumerate(md_files):
        docs.append(f.read_text(encoding="utf-8"))
        ids.append(f"kb_{i}")
        metas.append({"filename": f.name})
    kb.add(documents=docs, ids=ids, metadatas=metas)
    print(f"Knowledge Specialist KB loaded: {len(docs)} articles from {KB_DIR}")
    return kb


KB = load_kb()

# -- IN-MEMORY TASK STORE ------------------------------------------------

tasks = {}  # task_id -> task object

# -- REQUEST/RESPONSE MODELS -----------------------------------------------

class TaskRequest(BaseModel):
    query: str                # the ticket query to research
    ticket_number: str = ""
    context: str = ""         # optional: triage results, category, etc.


class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: str

# -- AGENT CARD ENDPOINT ----------------------------------------------------

@app.get("/agent-card")
def agent_card():
    """A2A Agent Card -- describes what this agent can do."""
    return {
        "agent_id": "isdo-knowledge-specialist-v1",
        "name": "ISDO Knowledge Specialist",
        "description": "Deep knowledge base lookup and past-incident analysis for IT support tickets. "
                        "Called by the Resolution Agent when ChromaDB confidence is LOW.",
        "version": "1.0.0",
        "capabilities": ["kb_search", "past_incident_analysis", "resolution_drafting"],
        "input_schema": {
            "query": "string -- describe the issue",
            "ticket_number": "string -- optional reference",
            "context": "string -- optional triage context (category, priority)",
        },
        "output_schema": {
            "articles_found": "int",
            "best_match": "string",
            "resolution": "string",
            "confidence": "HIGH | MEDIUM | LOW",
            "escalate_to_l2": "bool",
        },
        "endpoint": "http://localhost:8001",
        "task_endpoint": "/tasks",
    }

# -- TASK SUBMISSION ---------------------------------------------------------

@app.post("/tasks", response_model=TaskResponse)
async def create_task(req: TaskRequest):
    """Accept a knowledge lookup task and process it immediately (sync for demo simplicity)."""
    task_id = str(uuid.uuid4())[:8]

    results = KB.query(query_texts=[req.query], n_results=min(2, KB.count()))
    articles = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results.get("distances") else []

    best_article = metas[0].get("filename", "None") if metas else "None"
    best_content = articles[0] if articles else ""
    confidence_score = max(0.0, 1 - distances[0]) if distances else 0.0

    if confidence_score > 0.6:
        confidence, escalate = "HIGH", False
    elif confidence_score > 0.3:
        confidence, escalate = "MEDIUM", False
    else:
        confidence, escalate = "LOW", True

    if best_content:
        prompt = (f"You are a Senior IT Knowledge Specialist.\n"
                  f"Based on the KB article below, provide a detailed expert resolution for:\n"
                  f"Issue: {req.query}\nContext: {req.context}\n\n"
                  f"KB Article ({best_article}):\n{best_content}\n\n"
                  f"Provide: 1) Root cause analysis 2) Step-by-step fix 3) Prevention advice. "
                  f"Be specific and technical -- this goes to an L2 engineer.")
        llm_response = call_model(model=MODEL, max_tokens=500, messages=[{"role": "user", "content": prompt}])
        resolution = extract_text(llm_response)
    else:
        resolution = "No KB article found. Recommend manual investigation by L2 engineer."
        confidence, escalate = "LOW", True

    task_result = {
        "task_id": task_id,
        "ticket_number": req.ticket_number,
        "query": req.query,
        "status": "completed",
        "created_at": datetime.now().isoformat(),
        "result": {
            "articles_found": len(articles),
            "best_match": best_article,
            "confidence_score": round(confidence_score, 2),
            "confidence": confidence,
            "resolution": resolution,
            "escalate_to_l2": escalate,
        },
    }
    tasks[task_id] = task_result

    print(f"[A2A] Task {task_id}: '{req.query[:50]}' -> {confidence} ({confidence_score:.0%})")
    return TaskResponse(task_id=task_id, status="completed", created_at=task_result["created_at"])

# -- TASK RESULT RETRIEVAL ---------------------------------------------------

@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Retrieve the result of a completed task."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@app.get("/health")
def health():
    return {"status": "ok", "agent": "Knowledge Specialist", "kb_articles": KB.count()}

# -- DEMO: CALL THIS SERVER FROM THE COMMAND LINE ----------------------------
# PowerShell:
#   $body = @{ query = "VPN not connecting after password reset"; ticket_number = "INC0001001" } | ConvertTo-Json
#   $result = Invoke-RestMethod -Uri http://localhost:8001/tasks -Method Post -ContentType "application/json" -Body $body
#   Invoke-RestMethod -Uri "http://localhost:8001/tasks/$($result.task_id)"
#
# bash/macOS/Linux:
#   curl -X POST http://localhost:8001/tasks \
#     -H "Content-Type: application/json" \
#     -d '{"query": "VPN not connecting after password reset", "ticket_number": "INC0001001"}'