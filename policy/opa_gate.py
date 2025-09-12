from __future__ import annotations
import json, subprocess, os
from typing import Any, Dict

def opa_allow(input_doc: Dict[str, Any]) -> bool:
    if os.getenv("OPA_ENABLE") != "true":
        return True
    try:
        p = subprocess.run(
            ["opa", "eval", "-f", "json", "-d", "policies/", "data.agent.allow"],
            input=json.dumps({"input": input_doc}), text=True, capture_output=True, check=False
        )
        if p.returncode != 0:
            return False
        data = json.loads(p.stdout)
            
        return bool(data["result"][0]["expressions"][0]["value"])
    except Exception:
        return False
