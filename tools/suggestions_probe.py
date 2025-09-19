from fastapi.testclient import TestClient
import json
import os
import sys
from pathlib import Path

# Ensure test tokens are allowed
os.environ.setdefault("ALLOW_TEST_TOKENS", "1")
os.environ.setdefault("TEST_API_KEY", "test")

# Add workspace root to sys.path so "api" package can be imported when run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from api.app import app  # noqa: E402

client = TestClient(app)
headers = {"X-API-Key": os.getenv("TEST_API_KEY", "test")}

resp = client.post("/suggest/auto", headers=headers)
print("/suggest/auto:")
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))

# Generate a few targeted suggestions
goals = [
	"Typisierung und Linting verbessern (Pylance-Warnungen reduzieren)",
	"RBAC auf weitere Endpunkte ausweiten",
	"Öffentliche Read-only-Ansicht vorbereiten",
]
print("\n/suggest/generate (targeted goals):")
for g in goals:
	r = client.post("/suggest/generate", headers=headers, json={"goal": g, "focus_paths": ["api/app.py", "tests/"]})
	print(f"- Goal: {g}")
	print(json.dumps(r.json(), ensure_ascii=False, indent=2))

# List current suggestions (summary)
resp2 = client.get("/suggest/list", headers=headers)
print("\n/suggest/list:")
print(json.dumps(resp2.json(), ensure_ascii=False, indent=2))
