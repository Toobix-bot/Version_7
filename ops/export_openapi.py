import os, sys, yaml

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.app import app  # noqa: E402

# Generate OpenAPI dict via FastAPI (ensures current routes & models)
openapi_schema = app.openapi()
# Adjust servers dynamically from env or fallback
server_url = os.environ.get('PUBLIC_SERVER_URL', 'https://echo-realm.onrender.com')
openapi_schema['servers'] = [{"url": server_url, "description": "Public server"}]
# Ensure security scheme present
components = openapi_schema.setdefault('components', {}).setdefault('securitySchemes', {})
components['ApiKeyHeader'] = {"type": "apiKey", "in": "header", "name": "X-API-Key"}
# Default global security
openapi_schema['security'] = [{"ApiKeyHeader": []}]
# Write YAML
os.makedirs('docs', exist_ok=True)
with open('docs/openapi.yaml', 'w', encoding='utf-8') as f:
    yaml.safe_dump(openapi_schema, f, sort_keys=False, allow_unicode=True)
print('Wrote docs/openapi.yaml (servers url =', server_url, ')')
