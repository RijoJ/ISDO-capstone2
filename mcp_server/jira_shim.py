"""
ISDO Lab C2 - Mock Jira Service Management REST API (Flask Shim)
Mimics the Jira REST API for service requests so agents can make
real HTTP calls without touching production.

Endpoints:
  GET  /rest/agile/1.0/board/requests   - list all requests (filters: request_type, priority, assignee, status)
  GET  /rest/api/2/issue/<key>          - get one request, Jira-style nested 'fields'
  PUT  /rest/api/2/issue/<key>          - update a request in memory
  GET  /health                         - service status

Run with: python jira_shim.py   (default port 5002)
"""

from flask import Flask, jsonify, request
import csv
import os

app = Flask(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "requests.csv")
FILTER_FIELDS = ["request_type", "priority", "assignee", "status"]


def load_requests():
    data = {}
    try:
        with open(DATA_FILE, newline="") as f:
            for row in csv.DictReader(f):
                data[row["key"]] = dict(row)
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return data


REQUESTS = load_requests()


def to_fields(req):
    """Shape a flat CSV row into Jira's nested 'fields' object."""
    return {
        "summary": req.get("summary"),
        "priority": {"name": req.get("priority")},
        "status": {"name": req.get("status")},
        "assignee": {"displayName": req.get("assignee")},
        "customfield_sla": req.get("sla"),
        "issuetype": {"name": req.get("request_type")},
    }


@app.route("/rest/agile/1.0/board/requests", methods=["GET"])
def list_requests():
    """Return all service requests, optionally filtered."""
    results = list(REQUESTS.values())
    for field in FILTER_FIELDS:
        val = request.args.get(field)
        if val:
            results = [r for r in results if r.get(field, "").lower() == val.lower()]
    return jsonify({"issues": results, "total": len(results)})


@app.route("/rest/api/2/issue/<key>", methods=["GET"])
def get_request(key):
    """Return a single request by key, Jira-style."""
    req = REQUESTS.get(key)
    if not req:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    return jsonify({"key": key, "fields": to_fields(req)})


@app.route("/rest/api/2/issue/<key>", methods=["PUT"])
def update_request(key):
    """Update a service request in memory."""
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"errorMessages": ["No update body provided"]}), 400
    fields = data.get("fields", data)
    REQUESTS[key].update(fields)
    print(f"[Jira Mock] Updated {key}: {fields}")
    return jsonify({"key": key, "message": "Updated successfully"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Jira Mock", "requests_loaded": len(REQUESTS)})


if __name__ == "__main__":
    print("Jira Mock API starting on http://localhost:5002")
    print(f"Loaded {len(REQUESTS)} requests from {DATA_FILE}")
    print("Endpoints: GET /rest/agile/1.0/board/requests  |  GET /health")
    app.run(port=5002, debug=True)