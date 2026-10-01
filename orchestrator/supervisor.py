"""
ISDO Lab C6 - LangGraph Orchestrator
Wires Triage -> Resolution -> SLA -> [HITL] -> Communication into one StateGraph.

Guardrail: nodes compute their own risk/confidence numbers in code (not the
model), so routing decisions (HITL gate, auto-resolve) can't be talked around
by a model's text output.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

import anthropic
import chromadb
import requests
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # keep output printable on Windows

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("ANTHROPIC_API_KEY is not set. Put it in a .env file in the project root.")

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
TEMPERATURE = 0.0
SLA_MINUTES = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
SIMULATED_NOW = datetime(2024, 1, 15, 10, 30)
# Priorities allowed to auto-resolve at HIGH confidence. P1 never auto-resolves.
AUTO_RESOLVE_PRIORITIES = {"P2", "P3", "P4"}

# Lab C8: A2A Knowledge Specialist, called from resolution_node when ChromaDB confidence is LOW.
A2A_BASE_URL = os.environ.get("A2A_KNOWLEDGE_SPECIALIST_URL", "http://localhost:8001")
A2A_TIMEOUT_SECONDS = 15

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"

client = anthropic.Anthropic(api_key=API_KEY)
_temperature_supported = True

# -- STATE SCHEMA (exactly the fields from the prompt) -------------------------

class TicketState(TypedDict, total=False):
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str  # added in Lab C7: 'Access Grant' etc., needed by the access-grant HITL trigger
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    kb_article: str
    resolution_text: str
    auto_resolve: bool
    confidence: str
    sla_breach_risk: str
    escalation_required: bool
    hitl_required: bool
    hitl_reason: str  # added in Lab C7: which trigger fired, shown in the approval prompt
    hitl_approved: bool
    user_message: str
    final_status: str
    audit_log: list

# -- AUDIT LOG ------------------------------------------------------------

def log(agent, action, detail):
    entry = {"timestamp": datetime.now().isoformat(), "agent": agent, "action": action, "detail": detail}
    print(f"  [AUDIT] {agent}: {action} -- {detail}")
    return entry


def call_model(**kwargs):
    """Call the model with temperature=0.0; if the SDK/model rejects that parameter, retry once without it.

    Two different failures can happen here depending on the installed anthropic
    SDK version and the model: an older/newer SDK can raise a plain TypeError
    for an argument it doesn't recognise, while the API itself raises
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

# -- CHROMADB KB (same chunking as Lab C1) -------------------------------------

def chunk_article(text, filename):
    chunks, current_lines, current_heading = [], [], "Introduction"
    for line in text.split("\n"):
        if line.startswith("## ") and current_lines:
            chunks.append({"content": "\n".join(current_lines).strip(), "heading": current_heading, "filename": filename})
            current_lines = []
            current_heading = line[3:].strip()
        current_lines.append(line)
    if current_lines:
        chunks.append({"content": "\n".join(current_lines).strip(), "heading": current_heading, "filename": filename})
    return chunks


def build_kb():
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        sys.exit(f"No .md files found in {KB_DIR}.")
    db = chromadb.Client()
    try:
        db.delete_collection("isdo_kb")
    except Exception:
        pass
    kb = db.create_collection("isdo_kb", metadata={"hnsw:space": "cosine"})
    docs, ids, metas = [], [], []
    for f in md_files:
        for chunk in chunk_article(f.read_text(encoding="utf-8"), f.name):
            docs.append(chunk["content"])
            ids.append(f"kb_{len(ids)}")
            metas.append({"filename": f.name, "heading": chunk["heading"]})
    kb.add(documents=docs, ids=ids, metadatas=metas)
    print(f"KB loaded: {len(docs)} chunks from {len(md_files)} articles")
    return kb


KB = build_kb()

# ══════════════════════════════════════════════════════
# NODE 1 - TRIAGE (calls Claude to classify the ticket as JSON)
# ══════════════════════════════════════════════════════

def triage_node(state: TicketState) -> TicketState:
    print(f"\n> TRIAGE AGENT - {state['ticket_number']}")

    prompt = (f"Classify this IT ticket:\n\nSummary: {state['short_description']}\nDetails: {state['description']}\n\n"
              "Return JSON with: category, priority, assignment_group, pii_detected, reasoning. "
              "Category must be exactly ONE of: Network, Application, Hardware, Access, Email, Server, Software. "
              "Priority: P1=critical/many users, P2=significant, P3=single user, P4=request. "
              "Only return the JSON object, no other text.")

    response = call_model(model=MODEL, max_tokens=400, messages=[{"role": "user", "content": prompt}])
    text = extract_text(response)

    try:
        if "```" in text:
            text = text.split("```")[1].replace("json", "", 1).strip()
        result = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        result = json.loads(match.group(0)) if match else None
        if result is None:
            print(f"  (JSON parse failed, raw output: {text[:200]!r} -- using CSV category/priority as fallback)")
            result = {"category": state.get("category", "Unknown"), "priority": state.get("priority", "P3"),
                      "assignment_group": "Service-Desk", "pii_detected": False}

    audit_log = state.get("audit_log", [])
    audit_log.append(log("TriageAgent", "classify_ticket", f"{result.get('category')}/{result.get('priority')}"))

    print(f"  Category: {result.get('category')}  Priority: {result.get('priority')}")
    print(f"  Assign To: {result.get('assignment_group')}  PII: {result.get('pii_detected')}")

    return {**state,
            "triage_category": result.get("category", state.get("category", "")),
            "triage_priority": result.get("priority", state.get("priority", "P3")),
            "triage_assignment_group": result.get("assignment_group", "Service-Desk"),
            "pii_detected": bool(result.get("pii_detected", False)),
            "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# NODE 2 - RESOLUTION (queries ChromaDB, drafts a resolution)
# ══════════════════════════════════════════════════════

def resolution_node(state: TicketState) -> TicketState:
    print("\n> RESOLUTION AGENT - searching KB")

    results = KB.query(query_texts=[state["short_description"]], n_results=min(10, KB.count()))
    best_per_file = {}
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        fname = meta["filename"]
        if fname not in best_per_file or distance < best_per_file[fname]:
            best_per_file[fname] = distance

    if best_per_file:
        article_name, distance = min(best_per_file.items(), key=lambda kv: kv[1])
        article_content = (KB_DIR / article_name).read_text(encoding="utf-8")
    else:
        article_name, distance, article_content = "none", 1.0, ""

    score = max(0.0, 1 - distance)
    confidence = "HIGH" if score > 0.6 else ("MEDIUM" if score >= 0.35 else "LOW")
    priority = state.get("triage_priority", state.get("priority", "P3"))
    auto_resolve = confidence == "HIGH" and priority in AUTO_RESOLVE_PRIORITIES

    if article_content:
        prompt = (f"Based on this KB article, draft a resolution for the user.\n\nKB Article ({article_name}):\n"
                  f"{article_content}\n\nTicket: {state['short_description']}\nWrite 3-4 numbered steps. Plain English.")
        response = call_model(model=MODEL, max_tokens=400, messages=[{"role": "user", "content": prompt}])
        resolution = extract_text(response)
    else:
        resolution = "No matching KB article was found. This ticket needs manual investigation by L2."

    audit_log = state.get("audit_log", [])

    # Lab C8: local ChromaDB confidence is LOW -> ask the A2A Knowledge Specialist
    # for a deeper look before giving up and routing straight to a human. If the
    # specialist is unreachable, we deliberately do NOT raise -- we keep the local
    # LOW-confidence result exactly as it was, which (via sla_node's confidence=="LOW"
    # check) is itself what sends this ticket to the HITL gate. So "fall back to
    # HITL" here means: on any A2A failure, just proceed with what ChromaDB already
    # gave us and let the existing LOW-confidence HITL trigger do its job.
    if confidence == "LOW":
        print("  -> Confidence LOW -- calling A2A Knowledge Specialist")
        try:
            post_resp = requests.post(
                f"{A2A_BASE_URL}/tasks",
                json={"query": state["short_description"], "ticket_number": state["ticket_number"],
                      "context": f"category={state.get('triage_category', state.get('category', ''))}, "
                                 f"priority={priority}"},
                timeout=A2A_TIMEOUT_SECONDS,
            )
            post_resp.raise_for_status()
            task_id = post_resp.json()["task_id"]
            print(f"  -> POST {A2A_BASE_URL}/tasks  task_id: {task_id}")

            get_resp = requests.get(f"{A2A_BASE_URL}/tasks/{task_id}", timeout=A2A_TIMEOUT_SECONDS)
            get_resp.raise_for_status()
            a2a_result = get_resp.json()["result"]
            print(f"  -> GET  {A2A_BASE_URL}/tasks/{task_id}")

            confidence = a2a_result.get("confidence", confidence)
            score = a2a_result.get("confidence_score", score)
            resolution = a2a_result.get("resolution", resolution)
            article_name = a2a_result.get("best_match", article_name)
            auto_resolve = confidence == "HIGH" and priority in AUTO_RESOLVE_PRIORITIES

            print(f"  A2A RESULT: Best Match: {article_name}  |  Confidence: {confidence} ({score:.0%})")
            audit_log.append(log("ResolutionAgent", "a2a_knowledge_specialist",
                                 f"A2A returned {confidence} ({score:.0%}), article: {article_name}"))

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Knowledge Specialist isn't running / didn't respond in time -- this is
            # expected and handled, not a crash: keep the local LOW result and move on.
            print(f"  (A2A Knowledge Specialist unreachable: {type(e).__name__} -- "
                  f"keeping local LOW confidence, HITL will review this ticket)")
            audit_log.append(log("ResolutionAgent", "a2a_knowledge_specialist",
                                 f"A2A call failed ({type(e).__name__}) -- kept local LOW confidence"))

    audit_log.append(log("ResolutionAgent", "search_kb",
                         f"Article: {article_name}  Confidence: {confidence} ({score:.0%})  AutoResolve: {auto_resolve}"))

    print(f"  KB Article: {article_name}")
    print(f"  Confidence: {confidence} ({score:.0%})  |  Auto-resolve: {auto_resolve}")

    return {**state, "kb_article": article_name, "resolution_text": resolution,
            "auto_resolve": auto_resolve, "confidence": confidence, "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# NODE 3 - SLA (checks the deadline; sets hitl_required/hitl_reason for ANY of
# three triggers added across Labs C5-C7: P1 CRITICAL/BREACHED, LOW KB
# confidence, or an Access Grant request)
# ══════════════════════════════════════════════════════

def sla_node(state: TicketState) -> TicketState:
    print("\n> SLA AGENT - checking deadline")

    priority = state.get("triage_priority", state.get("priority", "P3"))
    category = state.get("triage_category", state.get("category", ""))
    try:
        due_dt = datetime.strptime(state.get("sla_due", ""), "%Y-%m-%d %H:%M:%S")
        minutes_remaining = int((due_dt - SIMULATED_NOW).total_seconds() / 60)
    except ValueError:
        minutes_remaining = 999

    total_minutes = SLA_MINUTES.get(priority, 480)
    if minutes_remaining < 0:
        risk = "BREACHED"
    elif minutes_remaining < total_minutes * 0.2:
        risk = "CRITICAL"
    elif minutes_remaining < total_minutes * 0.5:
        risk = "AT_RISK"
    else:
        risk = "ON_TRACK"

    escalation_required = risk in ("BREACHED", "CRITICAL") and priority in ("P1", "P2")

    # Lab C7: three independent HITL triggers. Checked in this priority order
    # so that when more than one applies, the most safety-critical reason is
    # the one shown to the approver (a P1 breach matters more than a low KB
    # score, even if both happen to be true for the same ticket).
    # Security-sensitive gate: trust the ticketing system's own category (what the
    # requester/portal actually filed this as) over the model's reclassification --
    # a ticket that mentions "VPN access" can get re-triaged as category=Network,
    # which must never be able to silently bypass the access-grant approval gate.
    is_access_grant = (state.get("category") == "Access" or category == "Access") \
        and state.get("request_type") == "Access Grant"
    is_low_confidence = state.get("confidence") == "LOW"
    is_p1_critical = priority == "P1" and risk in ("BREACHED", "CRITICAL")

    if is_p1_critical:
        hitl_required, hitl_reason = True, f"P1 SLA {risk} -- requires senior approval before escalation"
    elif is_access_grant:
        hitl_required, hitl_reason = True, "ACCESS GRANT -- contractor/new-hire access request requires security approval"
    elif is_low_confidence:
        hitl_required, hitl_reason = True, "LOW KB confidence -- Resolution Agent could not find a clear fix"
    else:
        hitl_required, hitl_reason = False, ""

    audit_log = state.get("audit_log", [])
    audit_log.append(log("SLAAgent", "get_sla_status", f"Risk: {risk}  Minutes remaining: {minutes_remaining}"))
    if hitl_required:
        audit_log.append(log("SLAAgent", "hitl_trigger", hitl_reason))

    print(f"  SLA Risk: {risk}  |  Minutes remaining: {minutes_remaining}")
    print(f"  Escalation required: {escalation_required}  |  HITL required: {hitl_required}")
    if hitl_required:
        print(f"  HITL reason: {hitl_reason}")

    return {**state, "sla_breach_risk": risk, "escalation_required": escalation_required,
            "hitl_required": hitl_required, "hitl_reason": hitl_reason, "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# NODE 4 - HITL GATE (asks a human for approval via input())
# ══════════════════════════════════════════════════════

def hitl_node(state: TicketState) -> TicketState:
    print("\n> HITL GATE - human approval required")
    print(f"  {'!!! ' * 8}")
    print(f"  Ticket:  {state['ticket_number']}  |  Priority: {state.get('triage_priority')}")
    print(f"  Reason:  {state.get('hitl_reason') or 'escalation pending'}")
    print(f"  {'!!! ' * 8}")

    try:
        decision = input("  Approve action? [y/n]: ").strip().lower()
    except EOFError:
        decision = "n"  # no interactive input available -> fail safe, do not approve
        print("  n  (no input available -- defaulting to NOT approved)")
    approved = decision == "y"

    audit_log = state.get("audit_log", [])
    audit_log.append(log("HITLGate", "approval_decision", "APPROVED" if approved else "REJECTED"))

    print(f"  Decision: {'APPROVED' if approved else 'REJECTED'}")
    return {**state, "hitl_approved": approved, "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# NODE 5 - COMMUNICATION (drafts the user message)
# ══════════════════════════════════════════════════════

def communication_node(state: TicketState) -> TicketState:
    print("\n> COMMUNICATION AGENT - drafting user message")

    if state.get("hitl_required"):
        if state.get("hitl_approved"):
            if (state.get("hitl_reason") or "").startswith("ACCESS GRANT"):
                # Doc's C7 sample output uses access-grant-specific wording rather than
                # the generic escalation message; this is additive, the other branches
                # below are unchanged from C6.
                msg = (f"Dear Requester,\n\nYour access grant request {state['ticket_number']} has been approved "
                       f"by security and will be provisioned shortly.\n\nIT Support Team")
            else:
                msg = (f"Dear User,\n\nYour ticket {state['ticket_number']} has been escalated to our senior support "
                       f"team and is being worked on as a priority. We will update you shortly.\n\nIT Support Team")
            final_status = "ESCALATED"
        else:
            msg = (f"Dear User,\n\nYour ticket {state['ticket_number']} requires additional review before we can "
                   f"proceed. A senior team member will contact you shortly.\n\nIT Support Team")
            final_status = "PENDING_REVIEW"
    elif state.get("auto_resolve"):
        msg = (f"Dear User,\n\nRegarding your ticket {state['ticket_number']}:\n\n{state.get('resolution_text', '')}\n\n"
               f"Please try these steps and let us know if the issue persists.\n\nIT Support Team")
        final_status = "RESOLVED"
    else:
        msg = (f"Dear User,\n\nYour ticket {state['ticket_number']} has been received and assigned to "
               f"{state.get('triage_assignment_group', 'our support team')}. We will keep you updated. "
               f"Reference: {state['ticket_number']}\n\nIT Support Team")
        final_status = "ESCALATED" if state.get("escalation_required") else "IN_PROGRESS"

    audit_log = state.get("audit_log", [])
    audit_log.append(log("CommunicationAgent", "post_comment", f"final_status={final_status}"))

    print(f"\n  USER MESSAGE:\n  {'-' * 40}\n  {msg}\n  {'-' * 40}")

    return {**state, "user_message": msg, "final_status": final_status, "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════

def route_after_sla(state: TicketState) -> Literal["hitl", "communication"]:
    return "hitl" if state.get("hitl_required") else "communication"

# ══════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(TicketState)
    g.add_node("triage", triage_node)
    g.add_node("resolution", resolution_node)
    g.add_node("sla", sla_node)
    g.add_node("hitl", hitl_node)
    g.add_node("communication", communication_node)

    g.set_entry_point("triage")
    g.add_edge("triage", "resolution")
    g.add_edge("resolution", "sla")
    g.add_conditional_edges("sla", route_after_sla, {"hitl": "hitl", "communication": "communication"})
    g.add_edge("hitl", "communication")
    g.add_edge("communication", END)
    return g.compile()


GRAPH = build_graph()

# ══════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    test_tickets = [
        {"ticket_number": "INC0001001", "short_description": "VPN not connecting after password change",
         "description": "VPN client fails after password reset. Auth failed error.",
         "category": "Network", "priority": "P2", "sla_due": "2024-01-15 14:00:00", "audit_log": []},
        {"ticket_number": "INC0001002", "short_description": "ERP system down - SAP login failing",
         "description": "Multiple Finance users cannot login to SAP. Error DBCON_FAIL.",
         "category": "Application", "priority": "P1", "sla_due": "2024-01-15 11:00:00", "audit_log": []},
        # Lab C7 Step 4: Access Grant request. HITL fires here from category + request_type
        # alone, independent of SLA risk -- note request_type is an addition beyond the doc's
        # literal test-ticket fields (see chat reply) because the access-grant HITL rule needs it.
        {"ticket_number": "REQ-1002", "short_description": "VPN access for new contractor",
         "description": "Contractor needs VPN access. Email: contractor@client.com",
         "category": "Access", "request_type": "Access Grant", "priority": "P2",
         "sla_due": "2024-01-15 15:00:00", "audit_log": []},
    ]

    all_results = []
    for ticket in test_tickets:
        print(f"\n{'=' * 55}\nPROCESSING TICKET: {ticket['ticket_number']}\n{'=' * 55}")
        result = GRAPH.invoke(ticket)
        all_results.append(result)
        print(f"\nFINAL STATUS: {result.get('final_status')}"
              + (f"  (HITL: {result.get('hitl_reason')})" if result.get("hitl_required") else ""))

    print(f"\n{'=' * 55}\nAUDIT LOG\n{'=' * 55}")
    for result in all_results:
        print(f"\n{result['ticket_number']}:")
        for entry in result.get("audit_log", []):
            print(f"  {entry['timestamp']}  {entry['agent']:<20}{entry['action']:<20}{entry['detail']}")