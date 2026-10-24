import io
import os
import tempfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.db import Base, get_session
from app.core.errors import TransientIngestionError
from app.main import app
from app.services import ingestion_service


@pytest_asyncio.fixture
async def engine():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    eng = create_async_engine(f"sqlite+aiosqlite:///{path}", future=True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()
    os.remove(path)


class JobRunner:
    """Stands in for the Celery queue (app.jobs.dispatch.enqueue_ingestion).

    inline=True (default): a job runs the moment it is enqueued, so a test that uploads
    and then queries sees a finished document, as it did before ingestion became async.
    inline=False: jobs pile up in `queue` and the test drives them with run_next() /
    run_all(), which is how the intermediate states (uploaded, processing) and the
    failure/redelivery cases are tested. `enqueue_error` simulates a broker outage.
    """

    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.inline = True
        self.queue: list[tuple[int, int]] = []
        self.results: list[str] = []
        self.transient_errors: list[Exception] = []
        self.enqueue_error: Exception | None = None

    async def enqueue(self, tenant_id: int, document_id: int) -> None:
        if self.enqueue_error is not None:
            raise self.enqueue_error
        if self.inline:
            await self.run(tenant_id, document_id)
        else:
            self.queue.append((tenant_id, document_id))

    async def run(self, tenant_id: int, document_id: int, **kwargs) -> str | None:
        try:
            result = await ingestion_service.process_document(
                self.session_factory, tenant_id, document_id, **kwargs
            )
        except TransientIngestionError as exc:  # a real worker would retry; tests inspect it
            self.transient_errors.append(exc)
            return None
        self.results.append(result)
        return result

    async def run_next(self) -> str | None:
        tenant_id, document_id = self.queue.pop(0)
        return await self.run(tenant_id, document_id)

    async def run_all(self) -> list[str | None]:
        return [await self.run_next() for _ in range(len(self.queue))]


@pytest_asyncio.fixture
async def jobs(engine, monkeypatch):
    from app.jobs import dispatch

    runner = JobRunner(async_sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(dispatch, "enqueue_ingestion", runner.enqueue)
    return runner


@pytest_asyncio.fixture
async def client(engine, jobs):
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_session():
        async with SessionLocal() as s:
            yield s

    app.dependency_overrides[get_session] = override_get_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Week 7: every endpoint but /ping and /auth/* needs a token. The
        # default `client` is the OWNER of a default tenant, so the pre-tenancy
        # tests keep exercising the same behaviour unchanged. Multi-tenant
        # tests build more principals with the `make_client` factory below.
        reg = await ac.post("/auth/register", json={
            "tenant_name": "Default Co", "email": "owner@default.test", "password": "password-123"})
        assert reg.status_code == 201, reg.text
        ac.headers["Authorization"] = "Bearer " + reg.json()["access_token"]
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def make_client(engine, jobs):
    """Factory for extra, independent callers (other tenants, other roles).
    Each is a separate AsyncClient with its own Authorization header, all
    sharing the same test database. Usage:
        owner_b = await make_client(register={"tenant_name": "B", "email": "b@b.test"})
        viewer  = await make_client(login=("v@x.test", "pw"))
        anon    = await make_client()
    """
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_session():
        async with SessionLocal() as s:
            yield s

    app.dependency_overrides[get_session] = override_get_session
    made = []

    async def factory(register: dict | None = None, login: tuple | None = None, token: str | None = None):
        ac = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        await ac.__aenter__()
        made.append(ac)
        if register is not None:
            body = {"password": "password-123", **register}
            r = await ac.post("/auth/register", json=body)
            assert r.status_code == 201, r.text
            token = r.json()["access_token"]
        elif login is not None:
            r = await ac.post("/auth/login", json={"email": login[0], "password": login[1]})
            assert r.status_code == 200, r.text
            token = r.json()["access_token"]
        if token:
            ac.headers["Authorization"] = "Bearer " + token
        return ac

    yield factory
    for ac in made:
        await ac.__aexit__(None, None, None)


@pytest_asyncio.fixture
async def session(engine):
    """
    A direct DB session for tests that need to inspect a column not
    exposed through the API (e.g. Chunk.embedding) -- the response schemas
    stay minimal on purpose, so this is the only way to assert on it.
    """
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with SessionLocal() as s:
        yield s


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """
    Replaces the real embedding provider call with a deterministic fake --
    same "no network call needed to run the test suite" philosophy as every
    other provider call mocked in this sprint. Returns one fixed-length
    zero vector per input text (length matches settings.embedding_dimensions),
    enough to prove embeddings were generated and stored without a real key.
    """
    from app.core.config import settings

    async def fake_embed_texts(texts):
        return [[0.0] * settings.embedding_dimensions for _ in texts]

    # Both call sites need patching -- ingestion AND retrieval each import
    # embed_texts separately (`from app.core.embeddings import embed_texts`
    # binds a new name in each importing module), so patching only one
    # leaves the other making a real model call. This was a real gap:
    # every retrieval-side query embedding was silently hitting the real
    # local model (and real network, to download it on first use) even
    # when this fixture was "mocking" embeddings -- only caught because a
    # network-restricted environment made it fail loudly instead of just
    # running slow.
    monkeypatch.setattr("app.services.ingestion_service.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.services.retrieval_service.embed_texts", fake_embed_texts)
    return fake_embed_texts


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """
    Replaces the real generation (chat) provider call with a deterministic
    fake -- same "no network call needed to run the test suite" philosophy
    as mock_embeddings above. Returns a fixed answer citing source [1],
    which is enough for most tests; tests that need a specific answer text
    (e.g. to check citation parsing) override this with their own
    monkeypatch.setattr call.
    """
    async def fake_get_chat_completion(messages):
        return "This is a mocked answer, grounded in the context [1]."

    monkeypatch.setattr(
        "app.services.generation_service.get_chat_completion", fake_get_chat_completion
    )
    return fake_get_chat_completion


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a real multi-page PDF; each inner list is the lines of one page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for lines in pages:
        y = 800
        for line in lines:
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


LONG_TEXT = " ".join(
    f"Sentence number {i} talks about topic {i % 5} in some detail." for i in range(40)
)


def _topic_vector(text: str, dim: int) -> list[float]:
    """
    A tiny hand-built embedding: puts a 1.0 in one of three "topic slots"
    based on keywords, everything else 0. Two texts about the same topic
    land on the SAME vector (distance 0); texts about different topics are
    orthogonal (distance 1). Deliberately crude -- real semantic nuance
    isn't the point, having a fake with predictable, checkable distances is.
    """
    vec = [0.0] * dim
    lowered = text.lower()
    if any(w in lowered for w in ("cat", "kitten", "feline")):
        vec[0] = 1.0
    elif any(w in lowered for w in ("revenue", "quarter", "finance")):
        vec[1] = 1.0
    else:
        vec[2] = 1.0
    return vec


def install_topic_embeddings(monkeypatch):
    """
    Overrides the default all-zero embedding mock (see mock_embeddings
    above) with the keyword-based fake, for tests that need distances to
    actually differ by content -- e.g. retrieval ranking tests. Patches
    both call sites (ingestion AND retrieval each import embed_texts
    separately), so a chunk embedded at upload time and a query embedded
    at retrieval time land on comparable vectors.
    """
    from app.core.config import settings

    async def fake_embed_texts(texts):
        return [_topic_vector(t, settings.embedding_dimensions) for t in texts]

    monkeypatch.setattr("app.services.ingestion_service.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.services.retrieval_service.embed_texts", fake_embed_texts)


class FakeLangChainChain:
    """
    Stands in for app.langchain_port.chain._chain (a LangChain
    RunnableSequence) in tests. RunnableSequence is a Pydantic model, so
    monkeypatching an `ainvoke` attribute directly onto an existing
    instance fails with "object has no field 'ainvoke'" -- Pydantic v2
    only allows assigning already-declared fields unless the model opts
    into extra="allow". Replacing the whole module-level `_chain`
    reference (see mock_langchain_llm and the tests below) sidesteps
    that restriction entirely, rather than fighting it.
    """

    def __init__(self, fn):
        self._fn = fn

    async def ainvoke(self, input_dict, config=None, **kwargs):
        return await self._fn(input_dict)


@pytest.fixture
def mock_langchain_llm(monkeypatch):
    """
    Replaces app.langchain_port.chain._chain wholesale with a
    FakeLangChainChain, same "no network call needed to run the test
    suite" philosophy as mock_llm/mock_embeddings above. NOT autouse --
    only tests/test_ask_langchain.py ever hits /ask/langchain.
    """
    from app.langchain_port import chain as chain_module

    async def fake_fn(input_dict):
        return "This is a mocked LangChain answer, grounded in the context [1]."

    monkeypatch.setattr(chain_module, "_chain", FakeLangChainChain(fake_fn))
    return fake_fn


@pytest.fixture
def scripted_agent(monkeypatch):
    """Install a ScriptedModel as the agent's chat model (Week 6). Returns
    the model so tests can inspect .calls / .tools_offered."""
    from app.agent import loop
    from tests.agent_helpers import ScriptedModel

    def install(turns: list) -> ScriptedModel:
        model = ScriptedModel(turns)

        def fake_get_model(tool_names):
            model.tools_offered.append(list(tool_names))
            return model

        monkeypatch.setattr(loop, "_get_model", fake_get_model)
        return model

    return install


@pytest.fixture(autouse=True)
def fast_password_hashing(monkeypatch):
    """scrypt at N=2**14 is deliberately slow (~50ms) -- right for logins,
    wasteful for a suite that registers hundreds of users. Hashes embed
    their own parameters, so this changes cost only, not correctness."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "password_hash_n", 2**4)


@pytest.fixture(autouse=True)
def fresh_rate_limiter():
    """Rate-limit counters are process-wide, and every test's database restarts tenant ids at 1:
    without this, one test's traffic would throttle the next test's 'tenant 1'."""
    from app.core import ratelimit

    ratelimit.reset()
    yield
    ratelimit.reset()
