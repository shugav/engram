#!/usr/bin/env python3
"""Engram v3.1.0 Stress Test Suite -- three tests, one report.

Purpose: Validate chunking, search quality, auto-connect, graph traversal,
and importance weighting at scale using real-world data sources.

Tests:
  1. Detective  -- Sherlock Holmes comprehension (chunking + recall on prose)
  2. Sprint     -- simulated multi-agent dev team (handoffs + cross-agent recall)
  3. Wikipedia  -- knowledge graph building (auto-connect + graph traversal)

Usage:
  # Create the stress DB (one time):
  docker exec engram-pg psql -U engram -d postgres -c "CREATE DATABASE engram_stress"

  # Run:
  ENGRAM_STRESS_DATABASE_URL="postgresql://engram:engram@127.0.0.1:5433/engram_stress" \\
    .venv/bin/python tests/stress_test.py

  # Cleanup after:
  .venv/bin/python tests/stress_test.py --cleanup
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engram.db import MemoryDB
from engram.embeddings import OllamaEmbedder
from engram.search import SearchEngine
from engram.types import Memory, MemoryType

logger = logging.getLogger("stress_test")

DATA_DIR = Path(__file__).parent / "stress_data"
REPORT_PATH = Path(__file__).parent / "stress_report.md"

SHERLOCK_BOOKS = [
    ("https://www.gutenberg.org/files/1661/1661-0.txt", "adventures"),
    ("https://www.gutenberg.org/files/834/834-0.txt", "memoirs"),
    ("https://www.gutenberg.org/files/108/108-0.txt", "return"),
    ("https://www.gutenberg.org/files/2852/2852-0.txt", "hound"),
    ("https://www.gutenberg.org/files/244/244-0.txt", "scarlet"),
]
GITHUB_REPOS = [
    "microsoft/vscode",
    "kubernetes/kubernetes",
    "facebook/react",
    "golang/go",
]
GITHUB_ISSUES_PER_REPO = 50

OLLAMA_MODEL = "snowflake-arctic-embed2"
OLLAMA_DIMS = 1024
OLLAMA_URL = "http://localhost:11434"

_EMBEDDER_CHOICE = "ollama"  # set by --embedder flag


def _embedder():
    """Create the embedding provider for this run."""
    if _EMBEDDER_CHOICE == "openai":
        from engram.embeddings import OpenAIEmbedder
        return OpenAIEmbedder()
    return OllamaEmbedder(
        base_url=OLLAMA_URL, model=OLLAMA_MODEL, dimensions=OLLAMA_DIMS,
    )


def _embedder_label() -> str:
    if _EMBEDDER_CHOICE == "openai":
        return "openai/text-embedding-3-small (1536 dims, API)"
    return f"{OLLAMA_MODEL} ({OLLAMA_DIMS} dims, Ollama local)"

WIKI_TOPICS = [
    # Pioneers
    "Alan_Turing", "John_von_Neumann", "Ada_Lovelace", "Charles_Babbage",
    "Grace_Hopper", "Dennis_Ritchie", "Ken_Thompson", "Linus_Torvalds",
    "Tim_Berners-Lee", "Vint_Cerf", "Claude_Shannon", "Donald_Knuth",
    "Margaret_Hamilton_(software_engineer)", "Steve_Wozniak",
    # Hardware
    "ENIAC", "Transistor", "Microprocessor", "Integrated_circuit",
    "Graphics_processing_unit", "DRAM", "Solid-state_drive",
    "Personal_computer",
    # Operating systems & languages
    "Unix", "Linux", "C_(programming_language)",
    "Python_(programming_language)", "Java_(programming_language)",
    "JavaScript", "Rust_(programming_language)", "Go_(programming_language)",
    "Lisp_(programming_language)", "Fortran", "COBOL",
    # Networking & web
    "ARPANET", "Internet", "World_Wide_Web", "Hypertext_Transfer_Protocol",
    "Domain_Name_System", "Transmission_Control_Protocol",
    # Concepts
    "Artificial_intelligence", "Machine_learning", "Cryptography",
    "Operating_system", "Database", "Algorithm", "Data_structure",
    "Compiler", "Open-source_software", "Version_control",
]


# ── Data Models ───────────────────────────────────────────────


@dataclass
class QueryResult:
    query: str
    expected_keywords: list[str]
    top_results: list[dict]
    hit_at_1: bool = False
    hit_at_3: bool = False
    reciprocal_rank: float = 0.0
    latency_ms: float = 0.0


@dataclass
class TestReport:
    name: str
    memories_stored: int = 0
    chunks_created: int = 0
    auto_connect_edges: int = 0
    avg_store_latency_ms: float = 0.0
    p95_store_latency_ms: float = 0.0
    avg_recall_latency_ms: float = 0.0
    p95_recall_latency_ms: float = 0.0
    hit_at_1_rate: float = 0.0
    hit_at_3_rate: float = 0.0
    mrr: float = 0.0
    avg_graph_score: float = 0.0
    avg_edges_per_memory: float = 0.0
    query_results: list[QueryResult] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────


def check_hit(content: str, keywords: list[str]) -> bool:
    """True if the content contains any of the expected keywords."""
    low = content.lower()
    return any(kw.lower() in low for kw in keywords)


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * 0.95), len(s) - 1)]


def score_queries(report: TestReport) -> None:
    """Compute aggregate search-quality metrics from individual query results."""
    n = len(report.query_results)
    if n == 0:
        return
    report.hit_at_1_rate = sum(q.hit_at_1 for q in report.query_results) / n
    report.hit_at_3_rate = sum(q.hit_at_3 for q in report.query_results) / n
    report.mrr = sum(q.reciprocal_rank for q in report.query_results) / n


async def run_queries(
    engine: SearchEngine,
    query_defs: list[dict],
    report: TestReport,
) -> None:
    """Run a list of {query, keywords} dicts and append QueryResults to the report."""
    latencies: list[float] = []
    for qdef in query_defs:
        t0 = time.monotonic()
        results = await engine.recall(qdef["query"], top_k=5)
        lat = (time.monotonic() - t0) * 1000
        latencies.append(lat)

        top = [
            {
                "id": r.memory.id[:8],
                "preview": r.memory.content[:120],
                "score": r.score,
                "chunk": (r.matched_chunk or "")[:120],
            }
            for r in results[:5]
        ]

        hit1 = bool(results) and check_hit(results[0].memory.content, qdef["keywords"])
        hit3 = any(check_hit(r.memory.content, qdef["keywords"]) for r in results[:3])
        rr = 0.0
        for i, r in enumerate(results[:5]):
            if check_hit(r.memory.content, qdef["keywords"]):
                rr = 1.0 / (i + 1)
                break

        report.query_results.append(QueryResult(
            query=qdef["query"],
            expected_keywords=qdef["keywords"],
            top_results=top,
            hit_at_1=hit1,
            hit_at_3=hit3,
            reciprocal_rank=rr,
            latency_ms=lat,
        ))

    report.avg_recall_latency_ms = sum(latencies) / len(latencies) if latencies else 0
    report.p95_recall_latency_ms = p95(latencies)
    score_queries(report)


async def collect_graph_stats(db: MemoryDB, report: TestReport) -> None:
    """Fill graph-quality fields on the report."""
    stats = await db.get_stats()
    report.memories_stored = stats.total_memories
    report.chunks_created = stats.total_chunks
    report.auto_connect_edges = stats.total_relationships

    scores: list[float] = []
    edges: list[int] = []
    for mem in await db.list_memories(limit=5000):
        scores.append(await db.get_graph_score(mem.id))
        edges.append(await db.get_connection_count(mem.id))
    report.avg_graph_score = sum(scores) / len(scores) if scores else 0
    report.avg_edges_per_memory = sum(edges) / len(edges) if edges else 0


# ── Data Fetching (download once, cache to disk) ──────────────


async def _cached_download(url: str, path: Path, label: str) -> str:
    if path.exists():
        logger.info("  cached: %s", path.name)
        return path.read_text(encoding="utf-8")
    logger.info("  downloading %s ...", label)
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text, encoding="utf-8")
    return r.text


def _split_gutenberg(raw: str, book_tag: str) -> list[dict]:
    """Split a Gutenberg text into story/chapter sections."""
    # Try roman-numeral chapter headers first (short story collections)
    parts = re.split(r"\n(?=[IVXLC]+\.\s+[A-Z]{2})", raw)
    stories = []
    for part in parts:
        text = part.strip()
        if len(text) < 500:
            continue
        title = text.split("\n", 1)[0].strip()
        stories.append({"title": f"[{book_tag}] {title}", "content": text[:20_000]})

    # Fallback for novels: split on "CHAPTER" headers
    if len(stories) <= 1:
        stories = []
        parts = re.split(r"\n(?=CHAPTER [IVXLC0-9]+)", raw)
        for part in parts:
            text = part.strip()
            if len(text) < 500:
                continue
            title = text.split("\n", 1)[0].strip()
            stories.append({"title": f"[{book_tag}] {title}", "content": text[:20_000]})

    # Last resort: fixed-size chunks
    if not stories:
        for i in range(0, len(raw), 8000):
            chunk = raw[i : i + 8000].strip()
            if len(chunk) > 300:
                stories.append({"title": f"[{book_tag}] Section {i // 8000 + 1}",
                                "content": chunk})
    return stories


async def fetch_sherlock() -> list[dict]:
    """Fetch all 5 Sherlock Holmes books and split into stories/chapters."""
    all_stories: list[dict] = []
    for url, tag in SHERLOCK_BOOKS:
        cache_name = f"sherlock_{tag}.txt"
        raw = await _cached_download(url, DATA_DIR / cache_name, f"Sherlock ({tag})")
        stories = _split_gutenberg(raw, tag)
        all_stories.extend(stories)
    logger.info("  total: %d stories/chapters across %d books", len(all_stories), len(SHERLOCK_BOOKS))
    return all_stories


async def fetch_github_issues() -> list[dict]:
    """Fetch issues from multiple popular repos."""
    cache = DATA_DIR / "github_issues_multi.json"
    if cache.exists():
        logger.info("  cached: %s", cache.name)
        return json.loads(cache.read_text())

    all_issues: list[dict] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for repo in GITHUB_REPOS:
            logger.info("  fetching %d issues from %s ...", GITHUB_ISSUES_PER_REPO, repo)
            try:
                resp = await client.get(
                    f"https://api.github.com/repos/{repo}/issues",
                    params={"state": "open", "sort": "comments", "direction": "desc",
                            "per_page": GITHUB_ISSUES_PER_REPO},
                    headers={"Accept": "application/vnd.github.v3+json"},
                )
                resp.raise_for_status()
                for item in resp.json():
                    if "pull_request" in item:
                        continue
                    all_issues.append({
                        "repo": repo,
                        "number": item["number"],
                        "title": item["title"],
                        "body": (item.get("body") or "")[:5000],
                        "labels": [lb["name"] for lb in item.get("labels", [])],
                        "comments": item.get("comments", 0),
                    })
            except httpx.HTTPStatusError as exc:
                logger.warning("  %s: HTTP %d (rate limit?)", repo, exc.response.status_code)
            await asyncio.sleep(1.0)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(all_issues, indent=2))
    logger.info("  cached %d issues across %d repos", len(all_issues), len(GITHUB_REPOS))
    return all_issues


async def fetch_wikipedia() -> list[dict]:
    """Return list of {title, topic_id, content} dicts."""
    cache = DATA_DIR / "wikipedia.json"
    if cache.exists():
        logger.info("  cached: %s", cache.name)
        return json.loads(cache.read_text())

    logger.info("  fetching %d Wikipedia articles ...", len(WIKI_TOPICS))
    articles: list[dict] = []
    headers = {"User-Agent": "EngramStressTest/1.0 (https://github.com/shugav/engram)"}
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for topic in WIKI_TOPICS:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "format": "json",
                        "titles": topic.replace("_", " "),
                        "prop": "extracts", "explaintext": True},
            )
            if resp.status_code == 200:
                pages = resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    ext = page.get("extract", "")
                    if ext:
                        articles.append({
                            "title": page.get("title", topic),
                            "topic_id": topic,
                            "content": ext[:20_000],
                        })
            await asyncio.sleep(0.25)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(articles, indent=2))
    logger.info("  cached %d articles", len(articles))
    return articles


# ── Database Setup ────────────────────────────────────────────


async def setup_pool(dsn: str):
    """Create DB if needed, reset schema, return pool."""
    import psycopg
    import engram.pool as pool_mod

    base = dsn.rsplit("/", 1)[0] + "/postgres"
    db_name = dsn.rsplit("/", 1)[1].split("?")[0]

    try:
        with psycopg.connect(base, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
                if not cur.fetchone():
                    cur.execute(f'CREATE DATABASE "{db_name}"')
                    logger.info("created database %s", db_name)
    except Exception as exc:
        logger.warning("db create: %s", exc)

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
            cur.execute("GRANT ALL ON SCHEMA public TO PUBLIC")

    os.environ["DATABASE_URL"] = dsn
    pool_mod._pool = None
    return await pool_mod.get_pool()


# ── Test 1: Detective ─────────────────────────────────────────

DETECTIVE_QUERIES = [
    # Adventures
    {"query": "Irene Adler woman outsmarted Holmes",
     "keywords": ["Irene", "Adler", "photograph"]},
    {"query": "red-headed league bank robbery tunnel",
     "keywords": ["red-headed", "Wilson", "tunnel", "cellar"]},
    {"query": "speckled band snake murder ventilator",
     "keywords": ["speckled", "band", "snake", "Roylott", "ventilator"]},
    {"query": "five orange pips threat letter",
     "keywords": ["orange", "pips", "Openshaw"]},
    {"query": "blue carbuncle goose Christmas gem",
     "keywords": ["carbuncle", "goose", "gem", "Christmas"]},
    {"query": "engineer thumb hydraulic press",
     "keywords": ["thumb", "hydraulic", "Hatherley"]},
    {"query": "beryl coronet banker stolen gems",
     "keywords": ["beryl", "coronet", "Holder"]},
    {"query": "copper beeches governess child hair",
     "keywords": ["copper", "beeches", "Hunter", "Rucastle"]},
    # Memoirs
    {"query": "silver blaze racehorse disappearance",
     "keywords": ["Silver Blaze", "horse", "Colonel"]},
    {"query": "Moriarty criminal mastermind Napoleon of crime",
     "keywords": ["Moriarty", "criminal", "Napoleon"]},
    {"query": "Greek interpreter kidnapping Mycroft",
     "keywords": ["Greek", "interpreter", "Mycroft"]},
    {"query": "Musgrave ritual treasure crown butler",
     "keywords": ["Musgrave", "ritual", "crown", "butler"]},
    # Return
    {"query": "empty house airgun assassination Moran",
     "keywords": ["empty house", "air-gun", "Moran", "wax"]},
    {"query": "dancing men cipher secret code",
     "keywords": ["dancing", "cipher", "code", "figures"]},
    {"query": "six napoleons busts pearl Borgia",
     "keywords": ["Napoleon", "bust", "pearl"]},
    # Hound
    {"query": "Baskerville hound curse moor fog",
     "keywords": ["Baskerville", "hound", "moor"]},
    {"query": "Stapleton naturalist villain Grimpen",
     "keywords": ["Stapleton", "Grimpen", "naturalist"]},
    # Scarlet
    {"query": "Study in Scarlet Jefferson Hope revenge poison",
     "keywords": ["Jefferson Hope", "poison", "revenge"]},
    {"query": "Mormon Utah desert Ferrier Lucy",
     "keywords": ["Mormon", "Utah", "Ferrier", "Lucy"]},
    # Cross-story
    {"query": "Holmes deduction disguise method observation",
     "keywords": ["Holmes", "Watson", "deduction"]},
]


async def test_detective(pool) -> TestReport:
    report = TestReport(name="Detective (Sherlock Holmes)")
    db = MemoryDB("stress-detective", pool)
    engine = SearchEngine(db=db, embedder=_embedder())

    stories = await fetch_sherlock()
    lats: list[float] = []
    for story in stories:
        t0 = time.monotonic()
        await engine.store(Memory(
            content=story["content"],
            memory_type=MemoryType.CONTEXT,
            tags=["sherlock", "story"],
            importance=2,
        ))
        lats.append((time.monotonic() - t0) * 1000)

    report.avg_store_latency_ms = sum(lats) / len(lats) if lats else 0
    report.p95_store_latency_ms = p95(lats)

    await run_queries(engine, DETECTIVE_QUERIES, report)
    await collect_graph_stats(db, report)

    # Findings
    if report.hit_at_1_rate >= 0.6:
        report.strengths.append(f"Hit@1 {report.hit_at_1_rate:.0%} on prose recall")
    else:
        report.weaknesses.append(
            f"Hit@1 only {report.hit_at_1_rate:.0%} -- paragraph chunking may not "
            "align with story-specific queries on long prose"
        )
    if report.auto_connect_edges > 0:
        report.strengths.append(
            f"Auto-connect linked {report.auto_connect_edges} edges across "
            f"{report.memories_stored} stories"
        )
    else:
        report.weaknesses.append(
            "No auto-connect edges -- embedding similarity may not cross "
            "the 0.78 threshold for distinct stories"
        )
    return report


# ── Test 2: Multi-Agent Sprint ────────────────────────────────

SPRINT_QUERIES = [
    # Session handoffs
    {"query": "session handoff agent status",
     "keywords": ["SESSION HANDOFF", "agent", "worked on"]},
    # VS Code topics
    {"query": "performance rendering slow lag",
     "keywords": ["performance", "rendering", "slow", "lag"]},
    {"query": "extension API plugin marketplace",
     "keywords": ["extension", "API", "plugin"]},
    {"query": "terminal shell command bash",
     "keywords": ["terminal", "shell", "command"]},
    {"query": "editor tabs workspace layout",
     "keywords": ["editor", "tab", "workspace"]},
    {"query": "accessibility screen reader keyboard",
     "keywords": ["accessibility", "screen reader", "keyboard"]},
    # Kubernetes topics
    {"query": "pod container crash restart",
     "keywords": ["pod", "container", "crash", "restart"]},
    {"query": "deployment rollout scaling replicas",
     "keywords": ["deployment", "rollout", "scaling", "replica"]},
    {"query": "node kubelet resource limits",
     "keywords": ["node", "kubelet", "resource", "limit"]},
    {"query": "service ingress load balancer networking",
     "keywords": ["service", "ingress", "load balancer", "network"]},
    # React topics
    {"query": "component rendering hooks state",
     "keywords": ["component", "render", "hook", "state"]},
    {"query": "concurrent mode suspense hydration",
     "keywords": ["concurrent", "suspense", "hydration"]},
    {"query": "performance profiler memo optimization",
     "keywords": ["performance", "profiler", "memo", "optimization"]},
    # Go topics
    {"query": "goroutine channel concurrency deadlock",
     "keywords": ["goroutine", "channel", "concurrency", "deadlock"]},
    {"query": "compiler gc garbage collection memory",
     "keywords": ["compiler", "gc", "garbage", "memory"]},
    # Cross-repo patterns
    {"query": "bug crash error exception failure",
     "keywords": ["bug", "crash", "error", "exception"]},
    {"query": "feature request enhancement proposal",
     "keywords": ["feature", "request", "enhancement"]},
    {"query": "documentation docs readme guide",
     "keywords": ["documentation", "docs", "readme", "guide"]},
    {"query": "security vulnerability CVE fix",
     "keywords": ["security", "vulnerability", "CVE"]},
    {"query": "test flaky CI pipeline failure",
     "keywords": ["test", "flaky", "CI", "pipeline"]},
]


async def test_sprint(pool) -> TestReport:
    report = TestReport(name="Multi-Agent Sprint (GitHub Issues)")
    db = MemoryDB("stress-sprint", pool)
    engine = SearchEngine(db=db, embedder=_embedder())

    issues = await fetch_github_issues()
    if not issues:
        report.findings.append("No GitHub issues fetched (API rate limit?)")
        return report

    lats: list[float] = []
    num_agents = 10
    batch = max(1, len(issues) // num_agents)

    for agent in range(num_agents):
        agent_issues = issues[agent * batch : (agent + 1) * batch]
        if not agent_issues:
            continue
        label_set = {lb for iss in agent_issues for lb in iss["labels"][:2]}
        repos_touched = {iss.get("repo", "unknown") for iss in agent_issues}

        issue_nums = ", ".join(f"#{i['number']}" for i in agent_issues[:5])
        topics = ", ".join(list(label_set)[:6])
        handoff = (
            f"SESSION HANDOFF: Agent {agent + 1} worked on "
            f"{len(agent_issues)} issues across {', '.join(repos_touched)}. "
            f"Issues: {issue_nums}. Topics: {topics}. "
            f"NEXT: Continue with remaining issues."
        )
        t0 = time.monotonic()
        await engine.store(Memory(
            content=handoff,
            memory_type=MemoryType.CONTEXT,
            tags=["session-handoff", f"agent-{agent + 1}"],
            importance=3,
        ))
        lats.append((time.monotonic() - t0) * 1000)

        for iss in agent_issues:
            repo = iss.get("repo", "unknown")
            body = (
                f"[{repo}] Issue #{iss['number']}: {iss['title']}\n"
                f"Labels: {', '.join(iss['labels'])}\n\n"
                f"{iss['body'][:4000]}"
            )
            is_bug = any("bug" in lb.lower() for lb in iss["labels"])
            t0 = time.monotonic()
            await engine.store(Memory(
                content=body,
                memory_type=MemoryType.ERROR if is_bug else MemoryType.CONTEXT,
                tags=["issue", f"agent-{agent + 1}", repo.split("/")[-1]]
                     + iss["labels"][:3],
                importance=2,
            ))
            lats.append((time.monotonic() - t0) * 1000)

    report.avg_store_latency_ms = sum(lats) / len(lats) if lats else 0
    report.p95_store_latency_ms = p95(lats)

    await run_queries(engine, SPRINT_QUERIES, report)
    await collect_graph_stats(db, report)

    # Extra check: can we find session handoffs specifically?
    handoff_results = await engine.recall("session handoff agent", top_k=10)
    found = sum(1 for r in handoff_results if "SESSION HANDOFF" in r.memory.content)
    if found >= 5:
        report.strengths.append(f"Session handoff recall: {found}/10 found in top 10")
    else:
        report.weaknesses.append(f"Session handoff recall weak: {found}/10 found")

    if report.auto_connect_edges > report.memories_stored * 0.3:
        report.strengths.append(
            f"Graph density: {report.auto_connect_edges} edges / "
            f"{report.memories_stored} memories"
        )
    else:
        report.weaknesses.append(
            f"Sparse graph: {report.auto_connect_edges} edges / "
            f"{report.memories_stored} memories"
        )
    return report


# ── Test 3: Wikipedia Knowledge Graph ─────────────────────────

WIKI_QUERIES = [
    # Pioneers
    {"query": "Turing machine computation theory",
     "keywords": ["Turing", "machine", "computation"]},
    {"query": "von Neumann stored program architecture",
     "keywords": ["von Neumann", "stored program", "EDVAC"]},
    {"query": "Ada Lovelace first programmer Babbage",
     "keywords": ["Ada", "Lovelace", "Babbage"]},
    {"query": "Claude Shannon information theory entropy",
     "keywords": ["Shannon", "information", "entropy"]},
    {"query": "Grace Hopper compiler COBOL Navy",
     "keywords": ["Hopper", "compiler", "COBOL"]},
    {"query": "Donald Knuth Art of Programming algorithms",
     "keywords": ["Knuth", "algorithm", "programming"]},
    # Hardware
    {"query": "ENIAC first electronic computer",
     "keywords": ["ENIAC", "electronic", "computer"]},
    {"query": "transistor semiconductor invention",
     "keywords": ["transistor", "semiconductor"]},
    {"query": "GPU graphics processing parallel computing",
     "keywords": ["GPU", "graphics", "parallel"]},
    {"query": "integrated circuit chip miniaturization",
     "keywords": ["integrated circuit", "chip"]},
    # OS & Languages
    {"query": "Unix operating system Bell Labs",
     "keywords": ["Unix", "Bell Labs", "operating system"]},
    {"query": "Linux kernel Torvalds open source",
     "keywords": ["Linux", "Torvalds", "kernel"]},
    {"query": "Python programming Guido van Rossum",
     "keywords": ["Python", "Guido", "programming"]},
    {"query": "Rust memory safety ownership borrow",
     "keywords": ["Rust", "memory safety", "ownership"]},
    {"query": "JavaScript web browser Netscape",
     "keywords": ["JavaScript", "browser", "web"]},
    # Networking
    {"query": "ARPANET internet network packet switching",
     "keywords": ["ARPANET", "internet", "network"]},
    {"query": "World Wide Web Berners-Lee HTML HTTP",
     "keywords": ["Web", "Berners-Lee", "HTML"]},
    {"query": "TCP IP protocol suite networking",
     "keywords": ["TCP", "protocol", "network"]},
    {"query": "DNS domain name resolution",
     "keywords": ["DNS", "domain", "name"]},
    # Concepts
    {"query": "artificial intelligence machine learning neural",
     "keywords": ["artificial intelligence", "machine learning"]},
    {"query": "cryptography encryption cipher security",
     "keywords": ["cryptography", "encryption", "cipher"]},
    {"query": "compiler source code machine translation",
     "keywords": ["compiler", "source code", "machine"]},
    {"query": "database relational SQL query",
     "keywords": ["database", "relational", "SQL"]},
    {"query": "version control git distributed",
     "keywords": ["version control", "git"]},
    {"query": "open source software free license community",
     "keywords": ["open source", "software", "license"]},
]


async def test_wiki(pool) -> TestReport:
    report = TestReport(name="Wikipedia Knowledge Graph (Computing History)")
    db = MemoryDB("stress-wiki", pool)
    engine = SearchEngine(db=db, embedder=_embedder())

    articles = await fetch_wikipedia()
    if not articles:
        report.findings.append("No Wikipedia articles fetched.")
        return report

    lats: list[float] = []
    for art in articles:
        tag = art["topic_id"].lower().replace("_", "-").replace("(", "").replace(")", "")
        t0 = time.monotonic()
        await engine.store(Memory(
            content=art["content"],
            memory_type=MemoryType.ARCHITECTURE,
            tags=["computing-history", "wikipedia", tag],
            importance=3,
        ))
        lats.append((time.monotonic() - t0) * 1000)

    report.avg_store_latency_ms = sum(lats) / len(lats) if lats else 0
    report.p95_store_latency_ms = p95(lats)

    await run_queries(engine, WIKI_QUERIES, report)
    await collect_graph_stats(db, report)

    # Graph traversal test: start at Turing, see what connects
    turing = await engine.recall("Alan Turing", top_k=1)
    if turing:
        reached = []
        for c in turing[0].connected:
            for kw in ["Unix", "Linux", "C ", "transistor", "von Neumann",
                        "ENIAC", "Babbage", "ARPANET"]:
                if kw.lower() in c.memory.content.lower():
                    reached.append(kw.strip())
                    break
        if reached:
            report.strengths.append(
                f"Graph traversal from Turing reaches: {', '.join(reached)}"
            )
        else:
            report.weaknesses.append(
                "Graph traversal from Turing reaches no related topics (1 hop)"
            )

    if report.auto_connect_edges > len(articles):
        report.strengths.append(
            f"Auto-connect built {report.auto_connect_edges} edges across "
            f"{len(articles)} articles ({report.avg_edges_per_memory:.1f}/memory)"
        )
    elif report.auto_connect_edges > 0:
        report.findings.append(
            f"Moderate connectivity: {report.auto_connect_edges} edges for "
            f"{len(articles)} articles"
        )
    else:
        report.weaknesses.append(
            "Zero auto-connect edges between related computing articles"
        )
    return report


# ── Report Generator ──────────────────────────────────────────


def generate_report(reports: list[TestReport], elapsed: float) -> str:
    L: list[str] = []

    def w(*args: str):
        L.extend(args)

    w(
        "# Engram v3.1.0 Stress Test Report", "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}  ",
        f"Runtime: {elapsed:.1f}s", "",
        "## Environment", "",
        f"- Embedder: {_embedder_label()}",
        f"- Database: Postgres 16 + pgvector 0.8.2",
        f"- Python: {sys.version.split()[0]}",
        "", "---", "",
    )

    all_s: list[str] = []
    all_w: list[str] = []

    for rpt in reports:
        w(f"## {rpt.name}", "")

        w("### Data", "",
          "| Metric | Value |", "|--------|-------|",
          f"| Memories | {rpt.memories_stored} |",
          f"| Chunks | {rpt.chunks_created} |",
          f"| Auto-connect edges | {rpt.auto_connect_edges} |",
          f"| Avg store latency | {rpt.avg_store_latency_ms:.0f} ms |",
          f"| P95 store latency | {rpt.p95_store_latency_ms:.0f} ms |", "")

        w("### Search Quality", "",
          "| Metric | Value |", "|--------|-------|",
          f"| Hit@1 | {rpt.hit_at_1_rate:.0%} |",
          f"| Hit@3 | {rpt.hit_at_3_rate:.0%} |",
          f"| MRR | {rpt.mrr:.3f} |",
          f"| Avg recall latency | {rpt.avg_recall_latency_ms:.0f} ms |",
          f"| P95 recall latency | {rpt.p95_recall_latency_ms:.0f} ms |", "")

        w("### Graph Quality", "",
          "| Metric | Value |", "|--------|-------|",
          f"| Avg graph score | {rpt.avg_graph_score:.4f} |",
          f"| Avg edges/memory | {rpt.avg_edges_per_memory:.2f} |", "")

        w("### Query Details", "",
          "| # | Query | Hit@1 | Hit@3 | RR | Latency |",
          "|---|-------|-------|-------|----|---------|")
        for i, q in enumerate(rpt.query_results, 1):
            h1 = "Y" if q.hit_at_1 else "-"
            h3 = "Y" if q.hit_at_3 else "-"
            w(f"| {i} | {q.query[:45]} | {h1} | {h3} | "
              f"{q.reciprocal_rank:.2f} | {q.latency_ms:.0f}ms |")
        w("")

        if rpt.strengths:
            w("### Strengths", "", *[f"- {s}" for s in rpt.strengths], "")
            all_s.extend(rpt.strengths)
        if rpt.weaknesses:
            w("### Weaknesses", "", *[f"- {x}" for x in rpt.weaknesses], "")
            all_w.extend(rpt.weaknesses)
        if rpt.findings:
            w("### Other Findings", "", *[f"- {f}" for f in rpt.findings], "")
        w("---", "")

    # ── Summary ───────────────────────────────────────────────

    tot_mem = sum(r.memories_stored for r in reports)
    tot_chk = sum(r.chunks_created for r in reports)
    tot_edg = sum(r.auto_connect_edges for r in reports)
    avg_h1 = sum(r.hit_at_1_rate for r in reports) / len(reports)
    avg_h3 = sum(r.hit_at_3_rate for r in reports) / len(reports)
    avg_mrr = sum(r.mrr for r in reports) / len(reports)

    w("## Summary", "",
      f"**Totals:** {tot_mem} memories, {tot_chk} chunks, {tot_edg} graph edges  ",
      f"**Search:** Hit@1 {avg_h1:.0%} | Hit@3 {avg_h3:.0%} | MRR {avg_mrr:.3f}", "")

    if all_s:
        w("### Overall Strengths", "", *[f"- {s}" for s in all_s], "")
    if all_w:
        w("### Overall Weaknesses", "", *[f"- {x}" for x in all_w], "")

    w("### Recommended Improvements", "")
    if avg_h1 < 0.7:
        w("- Chunking strategy may need tuning -- Hit@1 below 70%")
    if tot_edg < tot_mem:
        w("- Auto-connect threshold (0.78) may need tuning for this embedding model")
    if any(r.avg_recall_latency_ms > 500 for r in reports):
        w("- Recall latency above 500ms; batch graph score queries")
    if avg_mrr < 0.5:
        w("- MRR below 0.5 -- relevant results not consistently in top positions")
    w("- Compare results across different embedding models and thresholds")
    w("- Add cross-project recall tests (can sprint project find wiki memories?)")
    w("")

    # ── Methodology ───────────────────────────────────────────

    w(
        "## Methodology", "",
        "### Test Design", "",
        "Three independent stress tests targeting different Engram subsystems:", "",
        "1. **Detective (Sherlock Holmes)** -- Stores ~50 stories/chapters from 5 "
        "Sherlock Holmes books (Project Gutenberg, public domain). 20 queries test "
        "plot details, characters, cross-story and cross-book patterns.", "",
        "2. **Multi-Agent Sprint (GitHub Issues)** -- Fetches ~200 issues from 4 major "
        "open-source repos (VS Code, Kubernetes, React, Go). Simulates 10 agents each "
        "storing session handoffs and issue memories. 20 queries test cross-agent and "
        "cross-repo recall.", "",
        "3. **Wikipedia Knowledge Graph (Computing History)** -- Stores ~50 Wikipedia "
        "articles spanning pioneers, hardware, languages, networking, and CS concepts. "
        "25 queries test graph traversal and cross-domain connections.", "",
        "### Metrics", "",
        "- **Hit@1**: Top result contains expected keywords",
        "- **Hit@3**: Any of top 3 results contains expected keywords",
        "- **MRR**: Mean Reciprocal Rank of the first relevant result",
        "- **Latency**: Wall-clock ms per store/recall operation (avg + P95)",
        "- **Graph score**: count_factor * avg_strength (0 to 1)",
        "- **Auto-connect edges**: Relationships created automatically during store", "",
        "### Limitations", "",
        f"- Embedder: {OLLAMA_MODEL} via Ollama (local GPU inference).",
        "- Keyword matching for relevance is approximate; manual review of "
        "failing queries is recommended.",
        "- GitHub data depends on API rate limits (60 req/hr unauthenticated).",
        "- Results will differ significantly with a production embedding provider.", "",
    )

    return "\n".join(L)


# ── Main ──────────────────────────────────────────────────────


async def main():
    global _EMBEDDER_CHOICE
    ap = argparse.ArgumentParser(description="Engram stress test suite")
    ap.add_argument("--cleanup", action="store_true", help="Drop stress DB after run")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--embedder", choices=["ollama", "openai"], default="ollama",
                    help="Embedding provider (default: ollama)")
    args = ap.parse_args()
    _EMBEDDER_CHOICE = args.embedder

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    dsn = os.environ.get(
        "ENGRAM_STRESS_DATABASE_URL",
        "postgresql://engram:engram@127.0.0.1:5433/engram_stress",
    )

    logger.info("=== Engram v3.1.0 Stress Test ===")

    if not args.skip_download:
        logger.info("Phase 1: Data acquisition")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        await fetch_sherlock()
        await fetch_github_issues()
        await fetch_wikipedia()

    logger.info("Phase 2: Database setup")
    pool = await setup_pool(dsn)

    t_start = time.monotonic()
    reports: list[TestReport] = []

    logger.info("Phase 3a: Detective test")
    reports.append(await test_detective(pool))

    logger.info("Phase 3b: Sprint test")
    reports.append(await test_sprint(pool))

    logger.info("Phase 3c: Wikipedia test")
    reports.append(await test_wiki(pool))

    elapsed = time.monotonic() - t_start

    logger.info("Phase 4: Report")
    md = generate_report(reports, elapsed)
    REPORT_PATH.write_text(md)
    logger.info("Report: %s", REPORT_PATH)

    print("\n" + "=" * 60)
    print("STRESS TEST COMPLETE")
    print("=" * 60)
    for r in reports:
        print(f"\n  {r.name}:")
        print(f"    {r.memories_stored} memories, {r.chunks_created} chunks, "
              f"{r.auto_connect_edges} edges")
        print(f"    Hit@1 {r.hit_at_1_rate:.0%}  Hit@3 {r.hit_at_3_rate:.0%}  "
              f"MRR {r.mrr:.3f}")
        print(f"    Store {r.avg_store_latency_ms:.0f}ms  "
              f"Recall {r.avg_recall_latency_ms:.0f}ms")
    print(f"\n  Total: {elapsed:.1f}s")
    print(f"  Report: {REPORT_PATH}\n")

    if args.cleanup:
        logger.info("Phase 5: Cleanup")
        from engram.pool import close_pool
        await close_pool()
        import psycopg
        base = dsn.rsplit("/", 1)[0] + "/postgres"
        db_name = dsn.rsplit("/", 1)[1].split("?")[0]
        with psycopg.connect(base, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        logger.info("Dropped %s", db_name)
    else:
        from engram.pool import close_pool
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
