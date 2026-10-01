"""
ISDO Lab C9 - PII Redaction Middleware
Masks PII before any ticket data is sent to Claude.
Patterns covered: names (spaCy NER when available, regex fallback always),
email addresses, employee IDs, IP addresses, and phone numbers.

Usage:
    from guardrails.pii_redactor import redact, restore

    clean_text, mapping = redact(raw_text)
    # ... send clean_text to Claude ...
    original_text = restore(claude_response, mapping)
"""

import json
import re
from datetime import datetime

# Try to import spaCy -- graceful fallback if not installed
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("WARNING: spaCy not available -- using regex-only PII detection for everything "
          "except names, which fall back to a heuristic pattern (see NAME_PATTERN below).")

# -- REGEX PATTERNS ---------------------------------------------------------

PATTERNS = {
    "EMAIL":       r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
    "IP_ADDRESS":  r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "EMPLOYEE_ID": r'\b(?:EMP|ZEN|EMP-|ZEN-)\d{3,6}\b',
    "PHONE":       r'\b(?:\+91[\-\s]?)?\d{10}\b|\b\d{3}[\-\s]\d{3}[\-\s]\d{4}\b',
    "TICKET_REF":  r'\b(?:INC|REQ|CHG)\d{7}\b',   # keep ticket refs -- not PII
}

# Fallback for names when spaCy's NER isn't available (or as a safety net even
# when it is -- NER can miss unusual names). Matches runs of consecutive
# Title-Case words; WORD_STOPWORDS is then used to trim common IT-ticket
# nouns ("User", "Employee", "ID"...) off the ends of a run before deciding
# whether what's left looks like an actual name. This is a heuristic, not
# real NER: it can still miss single-word or lower-cased names and occasionally
# over- or under-trim an unusual phrase. See the install note at the bottom of
# this file for real NER via spaCy.
_WORD = r"[A-Z][a-zA-Z]*(?:['\u2019\-][A-Z][a-zA-Z]*)*"  # e.g. Smith, D'Souza, O'Brien, Jean-Pierre
NAME_RUN_PATTERN = rf"\b{_WORD}(?:\s+{_WORD})*\b"

WORD_STOPWORDS = {
    "user", "employee", "contractor", "customer", "requester", "account", "password",
    "ticket", "error", "system", "network", "access", "service", "desk", "support",
    "agent", "report", "request", "code", "status", "team", "group", "reset", "grant",
    "client", "contact", "manager", "admin", "specialist", "engineer", "department",
    "finance", "it", "hr", "ad", "vpn", "sap", "erp", "kb", "l1", "l2", "p1", "p2", "p3", "p4",
    "id", "emp", "zen", "inc", "req", "chg", "active", "directory", "exchange", "server",
    "no", "pii", "this", "part", "contractor", "project",
}


# -- AUDIT LOGGER -------------------------------------------------------------

audit_log = []


def _audit(action, detail):
    entry = {"timestamp": datetime.now().isoformat(), "module": "PIIRedactor", "action": action, "detail": detail}
    audit_log.append(entry)
    return entry

# -- REDACTION FUNCTION -------------------------------------------------------

def _trim_name_run(run_text):
    """Given a run of consecutive Title-Case words, strip stopword words off
    both ends and decide whether what's left is worth treating as a name.
    Returns (trimmed_text, offset_from_run_start) or (None, None) to reject."""
    words = run_text.split()
    start_idx, end_idx = 0, len(words)

    while start_idx < end_idx and words[start_idx].lower().strip("'\u2019-") in WORD_STOPWORDS:
        start_idx += 1
    while end_idx > start_idx and words[end_idx - 1].lower().strip("'\u2019-") in WORD_STOPWORDS:
        end_idx -= 1

    kept = words[start_idx:end_idx]
    if len(kept) < 2:
        return None, None  # a lone word is too risky to call a name
    if any(len(w) >= 2 and w.isupper() for w in kept):
        return None, None  # leftover acronym (e.g. "ZEN" after trimming "ID") -- but
        # allow a single uppercase letter through, e.g. a middle initial ("Michael D Souza")

    trimmed = " ".join(kept)
    offset = run_text.index(trimmed)
    return trimmed, offset


def redact(text: str) -> tuple[str, dict]:
    """
    Redact PII from text. Returns:
      - clean_text: text with PII replaced by tokens like [EMAIL_1], [NAME_1]
      - mapping: dict to restore original values later

    Example:
      clean, m = redact("Contact john.doe@corp.com or call 9876543210")
      # clean  = "Contact [EMAIL_1] or call [PHONE_1]"
      # m      = {"[EMAIL_1]": "john.doe@corp.com", "[PHONE_1]": "9876543210"}
    """
    # Collect every candidate as (start, end, label, matched_text), then
    # resolve overlaps and do a single reverse-order replace pass. This
    # avoids the bug in the original code, where sequential str.replace()
    # calls could redact the WRONG occurrence of a repeated substring, and
    # where a name match could swallow or be swallowed by an adjacent
    # EMPLOYEE_ID/other match.
    spans = []  # list of [start, end, label, text]

    if SPACY_AVAILABLE:
        doc = nlp(text)
        for ent in doc.ents:
            # Guard against a known spaCy false-positive: short ALL-CAPS acronyms
            # (PII, SLA, KB, VPN...) occasionally get tagged PERSON. Real names
            # in ticket text are essentially never written fully uppercase, so
            # this is a safe filter, not a real name being skipped.
            if ent.label_ == "PERSON" and not ent.text.isupper():
                spans.append([ent.start_char, ent.end_char, "NAME", ent.text])

    # Regex name fallback -- always runs (not just when spaCy is unavailable),
    # so a name spaCy's NER misses still gets caught. Only accepted where it
    # doesn't overlap a span spaCy already found.
    for match in re.finditer(NAME_RUN_PATTERN, text):
        trimmed, offset = _trim_name_run(match.group(0))
        if trimmed is None:
            continue
        start = match.start() + offset
        end = start + len(trimmed)
        if any(not (end <= s[0] or start >= s[1]) for s in spans):
            continue  # overlaps a spaCy-found span -- don't double-count it
        spans.append([start, end, "NAME", trimmed])

    # Other regex patterns (email, IP, employee ID, phone)
    for label, pattern in PATTERNS.items():
        if label == "TICKET_REF":
            continue  # Preserve ticket numbers -- not PII
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()
            if any(not (end <= s[0] or start >= s[1]) for s in spans):
                continue  # e.g. don't let EMAIL overlap a NAME span
            spans.append([start, end, label, match.group(0)])

    spans.sort(key=lambda s: s[0])

    # Number tokens in reading order (left to right), then build the output
    # by replacing from the end of the string backward so earlier offsets
    # stay valid as we go.
    mapping = {}
    counters = {}
    labeled_spans = []
    for start, end, label, matched_text in spans:
        counters[label] = counters.get(label, 0) + 1
        token = f"[{label}_{counters[label]}]"
        mapping[token] = matched_text
        labeled_spans.append((start, end, token))

    clean = text
    for start, end, token in sorted(labeled_spans, key=lambda s: s[0], reverse=True):
        clean = clean[:start] + token + clean[end:]

    pii_count = len(mapping)
    if pii_count > 0:
        _audit("redact", f"{pii_count} PII item(s) masked: {list(mapping.keys())}")
    else:
        _audit("redact", "No PII detected")

    return clean, mapping


def restore(text: str, mapping: dict) -> str:
    """Restore PII tokens back to original values (for system-of-record logging only)."""
    restored = text
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    _audit("restore", f"{len(mapping)} PII item(s) restored")
    return restored


def get_audit_log() -> list:
    """Return all PII redaction audit entries."""
    return audit_log

# -- AUDIT TRAIL LOGGER -------------------------------------------------------

class AuditLogger:
    """Logs every agent action with timestamp, agent name, tool, rationale, approval."""

    def __init__(self, log_file: str = "logs/audit_trail.jsonl"):
        import os
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.log_file = log_file
        self.entries = []

    def log(self, agent: str, action: str, ticket_number: str = "",
            tool: str = "", rationale: str = "", approval_status: str = "N/A"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "ticket_number": ticket_number,
            "tool": tool,
            "rationale": rationale[:200] if rationale else "",
            "approval_status": approval_status,
        }
        self.entries.append(entry)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"  [AUDIT] {agent} | {action} | {ticket_number} | {approval_status}")
        return entry

    def print_trail(self):
        print(f"\n{'=' * 55}")
        print(f"FULL AUDIT TRAIL ({len(self.entries)} entries)")
        print(f"{'=' * 55}")
        for e in self.entries:
            print(f"  {e['timestamp'][:19]}  {e['agent']:<22} {e['action']:<20} {e['approval_status']}")

# -- DEMO ----------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("PII REDACTION DEMO")
    print("=" * 55)

    sample_tickets = [
        "User John Smith (emp ID ZEN-9823) reports VPN failure. Contact: john.smith@zensar.com or +91-9876543210.",
        "Contractor sarah.jones@client.com needs access to REQ-1002. IP: 192.168.1.45.",
        "Password reset for Michael D'Souza. Employee EMP-00142. No PII in this part.",
        "VPN not connecting after password change. Error: authentication failed. Ticket INC0001001.",
    ]

    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Original : {ticket}")
        clean, mapping = redact(ticket)
        print(f"Redacted : {clean}")
        if mapping:
            print(f"Mapping  : {mapping}")

    print("\n" + "=" * 55)
    print("AUDIT TRAIL DEMO")
    print("=" * 55)

    logger = AuditLogger("logs/demo_audit.jsonl")
    logger.log("TriageAgent", "classify_ticket", "INC0001001", "classify_ticket",
               "Network/P2 -- VPN failure after password change", "Auto")
    logger.log("ResolutionAgent", "search_kb", "INC0001001", "search_kb",
               "KB article found: vpn_troubleshooting.md (85% confidence)", "Auto")
    logger.log("SLAAgent", "get_sla_status", "INC0001001", "get_sla_status",
               "SLA AT_RISK -- 210 min remaining of 240 min total", "Auto")
    logger.log("HITLGate", "approval_request", "INC0001002", "",
               "P1 escalation requires human approval", "PENDING")
    logger.log("HITLGate", "approval_decision", "INC0001002", "",
               "Human operator approved P1 escalation", "APPROVED")
    logger.log("CommunicationAgent", "post_comment", "INC0001001", "post_comment",
               "Resolution sent to user -- auto-resolved L1 ticket", "Auto")

    logger.print_trail()
    print(f"\nAudit log saved to: logs/demo_audit.jsonl")

# -- OPTIONAL: GET REAL NER INSTEAD OF THE REGEX FALLBACK --------------------
# The regex fallback above is a safety net, not a replacement for real NER --
# it will miss single-word names, lower-case names, and names that don't fit
# the "1-3 Title-Case words" shape, and can occasionally false-positive on an
# unanticipated Title-Case IT phrase. For production-quality name detection,
# install spaCy's small English model:
#   pip install spacy
#   python -m spacy download en_core_web_sm