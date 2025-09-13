import os
import sys
from pathlib import Path

# Ensure project root on path for 'api' imports when pytest changes CWD
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
import asyncio  # noqa: E402
from api.app import app  # noqa: E402

os.environ.setdefault("API_TOKENS", "test-token")

@pytest.fixture(scope="session")
def event_loop():  # Reuse a single loop so global asyncio.Queue in app stays bound
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest_asyncio.fixture
async def client():  # async test client fixture
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
