"""The knowledge folder: extraction, chunking, indexing, hybrid (embedding + keyword) search."""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ai_buddy.knowledge import (KnowledgeBase, chunk_document, embed_format, extract, fts_query, is_indexable, keywords,
                                min_similarity, pack, rrf, sections)

HENDERSON = """# Henderson project: talking points
- Open with the Henderson ops team story: they saved ten hours a week after switching to automated reports.
- Numbers: revenue up 12% quarter over quarter, churn down to 3%, NPS 61.
"""
BUDGET = """# Budget notes
Rent is 1,650 a month. Groceries target 500, eating out capped at 150.
"""
RUNNING = """# Training log
Long run on Sunday: 21 kilometres along the river at an easy pace. Knee felt fine.
"""


# ---------- extraction & chunking ----------

def test_markdown_sections_keep_the_heading_path():
    text = "intro\n# A\none\n## B\ntwo\n```\n# not a heading\n```\n# C\nthree"
    assert sections(text, True) == [("", "intro"), ("A", "one"), ("A › B", "two\n```\n# not a heading\n```"),
                                    ("C", "three")]
    assert sections(text, False) == [("", text)]


def test_pack_overlaps_and_splits_long_paragraphs():
    paras = [f"Paragraph {i} " + "word " * 40 for i in range(6)]
    chunks = pack(paras, 600, 100)
    assert len(chunks) > 1 and all(len(c) <= 700 for c in chunks)
    assert chunks[1].startswith("…")                       # carries the end of the previous chunk
    long = pack(["Sentence one. " * 200], 500, 0)
    assert len(long) > 3 and all(len(c) <= 500 for c in long)


def test_chunk_document_attaches_headings_and_pages():
    chunks = chunk_document([("Page one text.", 1), ("Page two text.", 2)], False)
    assert [(c.text, c.page) for c in chunks] == [("Page one text.", 1), ("Page two text.", 2)]
    md = chunk_document([(HENDERSON, None)], True)
    assert md[0].heading == "Henderson project: talking points"


def test_docx_and_html_extraction(tmp_path):
    docx = tmp_path / "plan.docx"
    body = ('<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Plan</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>Ship the </w:t></w:r><w:r><w:t>beta &amp; gather feedback.</w:t></w:r></w:p>')
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="w"><w:body>{body}</w:body></w:document>')
    (text, _), = extract(docx)[0]
    assert text == "# Plan\n\nShip the beta & gather feedback."
    html = tmp_path / "page.html"
    html.write_text("<html><head><title>T</title><script>var x</script></head><body><h1>Hello</h1><p>World</p></body></html>")
    (text, _), = extract(html)[0]
    assert "Hello" in text and "World" in text and "var x" not in text


@pytest.mark.parametrize("name,ok", [("notes.md", True), ("README", True), ("a.pdf", True), (".hidden.md", False),
                                     ("~$lock.docx", False), ("x.md~", False), ("photo.jpg", False)])
def test_is_indexable(name, ok):
    assert is_indexable(Path(name)) == ok


def test_query_helpers():
    assert fts_query("What did I write about the Henderson budget?") == '"henderson" OR "budget"'
    assert keywords("budget budget budget rent rent groceries", 2) == "budget rent"
    assert [i for i, _ in rrf([[(1, 9.0), (2, 8.0)], [(2, 0.9), (3, 0.8)]])] == [2, 1, 3]


# ---------- the index ----------

WORDS = ["henderson", "revenue", "churn", "talking", "points", "rent", "groceries", "budget", "money", "spend",
         "run", "river", "kilometres", "knee", "exercise", "pitch", "numbers"]
SYNONYMS = {"money": "budget", "spend": "budget", "exercise": "run", "pitch": "talking", "numbers": "revenue"}


class FakeEmbedder:
    """Bag of words with synonyms: enough "meaning" to tell semantic search from keyword search."""

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        if self.fail:
            raise RuntimeError("model not loaded")
        out = []
        for t in texts:
            v = np.zeros(len(WORDS), dtype=np.float32)
            for w in re.findall(r"[a-z]+", t.lower()):
                w = SYNONYMS.get(w, w)
                if w in WORDS:
                    v[WORDS.index(w)] += 1
            out.append((v + 1e-3).tolist())
        return out


def folder(tmp_path: Path) -> Path:
    notes = tmp_path / "notes"
    (notes / "work").mkdir(parents=True)
    (notes / "work" / "henderson.md").write_text(HENDERSON)
    (notes / "budget.md").write_text(BUDGET)
    (notes / "running.md").write_text(RUNNING)
    return notes


async def test_index_and_keyword_search(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db", folder(tmp_path), watch=False)
    await kb.reindex()
    st = kb.status()
    assert st["files"] == 3 and st["chunks"] == 3 and st["embeddings"] == "off"
    hits = await kb.search("Henderson talking points", 2)
    assert hits[0].file == "work/henderson.md" and hits[0].heading == "Henderson project: talking points"
    await kb.close()


async def test_reindex_picks_up_changes_and_deletions(tmp_path):
    notes = folder(tmp_path)
    kb = KnowledgeBase(tmp_path / "kb.db", notes, watch=False)
    await kb.reindex()
    (notes / "budget.md").unlink()
    (notes / "ideas.md").write_text("# Ideas\nA podcast about urban gardening.")
    os.utime(notes / "running.md", (1, 1))
    await kb.reindex()
    files = {h.file for h in await kb.search("podcast gardening budget rent", 5)}
    assert "ideas.md" in files and "budget.md" not in files
    await kb.set_folder(None)
    assert kb.status()["files"] == 0
    await kb.close()


async def test_hybrid_search_finds_meaning_not_just_words(tmp_path):
    emb = FakeEmbedder()
    kb = KnowledgeBase(tmp_path / "kb.db", folder(tmp_path), watch=False, embed=emb,
                       embed_model="Qwen3-Embedding-0.6B-GGUF")
    await kb.reindex()
    st = kb.status()
    assert st["embeddings"] == "ready" and st["embedded_chunks"] == 3 and st["embed_model"] == "Qwen3-Embedding-0.6B-GGUF"
    # No keyword overlap with the budget note ("money", "spend"), but the meaning matches.
    hits = await kb.search("how much money do I spend", 1)
    assert hits[0].file == "budget.md"
    # Documents are embedded as they are, queries with Qwen3's instruction.
    assert emb.calls[0][0].startswith("budget\n# Budget notes") or "\n" in emb.calls[0][0]
    assert emb.calls[-1][0].startswith("Instruct: Given a question about the user's own notes")
    await kb.close()


async def test_changing_the_embedding_model_re_embeds(tmp_path):
    notes = folder(tmp_path)
    kb = KnowledgeBase(tmp_path / "kb.db", notes, watch=False, embed=FakeEmbedder(), embed_model="nomic-embed-text-v1-GGUF")
    await kb.reindex()
    await kb.close()
    emb = FakeEmbedder()
    kb2 = KnowledgeBase(tmp_path / "kb.db", notes, watch=False, embed=emb, embed_model="Qwen3-Embedding-0.6B-GGUF")
    await kb2.reindex()
    assert sum(len(c) for c in emb.calls) == 3          # every chunk embedded again with the new model
    await kb2.reindex()
    assert sum(len(c) for c in emb.calls) == 3          # and not again while nothing changes
    await kb2.close()


async def test_without_the_embedding_model_search_falls_back_to_keywords(tmp_path):
    emb = FakeEmbedder(fail=True)
    kb = KnowledgeBase(tmp_path / "kb.db", folder(tmp_path), watch=False, embed=emb, embed_model="x")
    await kb.reindex()
    assert kb.status()["embeddings"].startswith("unavailable")
    hits = await kb.search("rent groceries", 1)
    assert hits[0].file == "budget.md"
    emb.fail = False
    await kb.reindex()                                   # tried again on the next scan
    assert kb.status()["embeddings"] == "ready"
    await kb.close()


async def test_strict_search_needs_meaning_not_a_shared_word(tmp_path):
    """Practice talking points: a resume that merely shares "learning" with a talk about sleep isn't one of them."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "resume.md").write_text("# Resume\nMachine learning engineer: Python, model training, deployment.")
    (notes / "sleep-talk.md").write_text("# Sleep talk: talking points\nDeep sleep keeps facts; REM links ideas.")

    async def embed(texts):           # "meaning": resume-like texts point one way, sleep-like texts the other
        return [[1.0, 0.05] if ("resume" in t.lower() or "python" in t.lower()) else [0.05, 1.0] for t in texts]
    kb = KnowledgeBase(tmp_path / "kb.db", notes, watch=False, embed=embed, embed_model="Qwen3-Embedding-0.6B-GGUF")
    await kb.reindex()
    query = "why sleep matters for learning talking points"
    assert {h.file for h in await kb.search(query, 4)} == {"resume.md", "sleep-talk.md"}     # "learning" matches
    assert [h.file for h in await kb.search(query, 4, strict=True)] == ["sleep-talk.md"]
    assert await kb.search("python resume", 4, strict=True) != []
    await kb.close()


async def test_passages_found_only_by_meaning_must_be_close_enough(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "resume.md").write_text("# Resume\nData analyst at Prairie Health.")

    async def embed(texts):           # everything is only weakly related (cosine 0.3)
        return [[1.0, 0.0] if t.startswith("Instruct") else [0.3, 0.954] for t in texts]
    kb = KnowledgeBase(tmp_path / "kb.db", notes, watch=False, embed=embed, embed_model="Qwen3-Embedding-0.6B-GGUF")
    await kb.reindex()
    assert kb.min_sim == min_similarity("Qwen3-Embedding-0.6B-GGUF") == 0.38
    assert await kb.search("what's in my notes about the budget?", 4) == []     # nothing in common, nothing found
    assert [h.file for h in await kb.search("Prairie Health", 4)] == ["resume.md"]   # the words still find it
    await kb.close()


def test_embedding_prompt_formats():
    q, d = embed_format("Qwen3-Embedding-0.6B-GGUF")
    assert q.format(query="budget?").endswith("\nQuery:budget?") and d.format(title="t", text="x") == "t\nx"
    q, d = embed_format("nomic-embed-text-v1-GGUF")
    assert q.format(query="budget?") == "search_query: budget?"
    assert d.format(title="t", text="x") == "search_document: t\nx"
    assert embed_format("something-else") == ("{query}", "{title}\n{text}")
