"""
ISDO Lab C2 - Mock ServiceNow REST API (Flask Shim)
Mimics the ServiceNow Table API so agents can make real HTTP calls
without touching a production system.

Endpoints:
  GET   /api/now/table/incident            - list all incidents (filters: category, priority, state, assignment_group)
  GET   /api/now/table/incident/<number>   - get one incident
  PATCH /api/now/table/incident/<number>   - update a row in memory
  GET   /health                            - service status

Run with: python snow_shim.py   (default port 5001)
"""

from flask import Flask, jsonify, request
import csv
import os

app = Flask(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "incidents.csv")
FILTER_FIELDS = ["category", "priority", "state", "assignment_group"]


def load_incidents():
    incidents = {}
    try:
        with open(DATA_FILE, newline="") as f:
            for row in csv.DictReader(f):
                incidents[row["number"]] = dict(row)
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return incidents


INCIDENTS = load_incidents()


@app.route("/api/now/table/incident", methods=["GET"])
def list_incidents():
    """Return all incidents, optionally filtered by query params."""
    results = list(INCIDENTS.values())
    for field in FILTER_FIELDS:
        val = request.args.get(field)
        if val:
            results = [r for r in results if r.get(field, "").lower() == val.lower()]
    return jsonify({"result": results, "total": len(results)})


@app.route("/api/now/table/incident/<number>", methods=["GET"])
def get_incident(number):
    """Return a single incident by number."""
    incident = INCIDENTS.get(number)
    if not incident:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": incident})


@app.route("/api/now/table/incident/<number>", methods=["PATCH"])
def update_incident(number):
    """Update fields on an incident (e.g. state, work_notes) in memory."""
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    updates = request.get_json(silent=True)
    if not updates:
        return jsonify({"error": "No update body provided"}), 400
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.route("/api/now/table/incident", methods=["POST"])
def create_incident():
    """Create a new incident."""
    data = request.get_json(silent=True)
    if not data or "number" not in data:
        return jsonify({"error": "Missing required field: number"}), 400
    INCIDENTS[data["number"]] = data
    print(f"[ServiceNow Mock] Created incident: {data['number']}")
    return jsonify({"result": data, "message": "Incident created"}), 201


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock", "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print("ServiceNow Mock API starting on http://localhost:5001")
    print(f"Loaded {len(INCIDENTS)} incidents from {DATA_FILE}")
    print("Endpoints: GET /api/now/table/incident  |  GET /health")
    app.run(port=5001, debug=True)