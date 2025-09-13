import os, sys, yaml

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.app import app  # noqa: E402

# Generate OpenAPI dict via FastAPI (ensures current routes & models)
openapi_schema = app.openapi()  # already includes servers, security, content-type fixes, hardening

# Ensure OpenAPI version and server URL are set as desired
openapi_schema['openapi'] = '3.0.3'
server_url = os.environ.get('PUBLIC_SERVER_URL') or (
    (openapi_schema.get('servers') or [{}])[0].get('url') if openapi_schema.get('servers') else None
) or 'https://version-7.onrender.com'
openapi_schema['servers'] = [{"url": server_url, "description": "Public server"}]

# Write YAML
os.makedirs('docs', exist_ok=True)
with open('docs/openapi.yaml', 'w', encoding='utf-8') as f:
    yaml.safe_dump(openapi_schema, f, sort_keys=False, allow_unicode=True)
print('Wrote docs/openapi.yaml (servers url =', server_url, ')')
