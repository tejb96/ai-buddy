"""Web helpers: how current a page is, and live questions putting the newest source first."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from ai_buddy import web
from ai_buddy.web import Page, is_live, page_time, parse_html, parse_when, read_pages

NOW = datetime(2026, 10, 14, 15, 30, tzinfo=ZoneInfo("America/Edmonton"))


@pytest.mark.parametrize("text,expected,timed", [
    ("2026-10-14T13:05:00-06:00", datetime(2026, 10, 14, 13, 5), True),
    ("2026-10-14T19:05:00Z", datetime(2026, 10, 14, 13, 5), True),             # UTC -> Edmonton (MDT)
    ("2026-10-12", datetime(2026, 10, 12), False),
    ("Calgary Int'l Airport 2:00 PM MDT Wednesday 14 October 2026", datetime(2026, 10, 14, 14, 0), True),
    ("Oct. 13, 2026 at 9:41 p.m.", datetime(2026, 10, 13, 21, 41), True),
    ("12 minutes ago", datetime(2026, 10, 14, 15, 18), True),
    ("2:15 pm", datetime(2026, 10, 14, 14, 15), True),
])
def test_parse_when(text, expected, timed):
    assert parse_when(text, NOW) == (expected, timed)


def test_a_bare_time_later_today_is_not_taken():
    assert parse_when("11:45 pm", NOW) is None


def test_page_time_prefers_the_pages_own_dates():
    text = "Observed at: Calgary Int'l Airport 2:00 PM MDT Wednesday 14 October 2026\nUpdated 5 minutes ago"
    assert page_time([], text, "Wed, 14 Oct 2026 21:29:00 GMT", NOW) == (datetime(2026, 10, 14, 15, 25), True)
    meta = ["2024-10-23T04:47:34Z", "2026-11-30"]                   # the article's date; a date in the future
    related = ["2026-10-14T21:45:47"]                               # a sidebar item, in UTC without saying so
    assert page_time(meta, "Related: updated 1 hour ago", None, NOW, related) == (datetime(2024, 10, 22, 22, 47), True)
    assert page_time([], "No dates here.", None, NOW, related) == (datetime(2026, 10, 14, 15, 45), True)
    assert page_time([], "No dates here.", "Tue, 13 Oct 2026 20:00:00 GMT", NOW) == (datetime(2026, 10, 13, 14, 0), True)
    assert page_time([], "Nothing.", None, NOW) == (None, False)


def test_parse_html_reads_meta_json_ld_and_time_tags():
    html = """<html><head><title>Flames win 4-2</title>
      <meta property="article:published_time" content="2026-10-13T22:10:00-06:00">
      <script type="application/ld+json">{"@type": "NewsArticle", "dateModified": "2026-10-14T08:30:00-06:00"}</script>
      </head><body><main><p>The Flames beat the Oilers 4-2 on Tuesday night at the Saddledome.</p></main></body></html>"""
    page = parse_html(html, NOW)
    assert page.title == "Flames win 4-2" and "Flames beat the Oilers" in page.text
    assert page.as_of == datetime(2026, 10, 14, 8, 30) and page.as_of_text() == "Wed Oct 14 2026, 8:30 AM"
    assert parse_html("<p>x</p>").as_of is None                   # without `now`, no date work


def test_live_questions():
    assert is_live("Calgary weather today") and is_live("who won the Flames game") and is_live("bitcoin price")
    assert not is_live("capital of Australia") and not is_live("how to boil an egg")


async def test_live_questions_put_the_newest_source_first(monkeypatch):
    pages = {"https://a.example": Page("A", "Calgary weather: 9 degrees and cloudy this morning, wind 20 km/h. " * 3,
                                       datetime(2026, 10, 14, 7, 0), True),
             "https://b.example": Page("B", "Calgary weather now: 16 degrees, feels like 12, sunny skies. " * 3,
                                       datetime(2026, 10, 14, 15, 0), True),
             "https://c.example": Page("C", "Calgary weather: average October temperatures are around 12. " * 3)}

    async def fake_fetch(http, url, timeout, now=None):
        return pages[url]
    monkeypatch.setattr(web, "fetch_page", fake_fetch)
    results = [{"title": k[8:], "url": k, "snippet": ""} for k in pages]
    read = await read_pages(None, results, "Calgary weather today", pages=2, chars=400, timeout=1, now=NOW)
    assert [r["url"] for r in read] == ["https://b.example", "https://a.example"]      # 3 tried, newest first
    assert read[0]["as_of_text"] == "Wed Oct 14 2026, 3:00 PM"
    other = await read_pages(None, results, "Calgary founding history", pages=2, chars=400, timeout=1, now=NOW)
    assert [r["url"] for r in other] == ["https://a.example", "https://b.example"]     # not live: search order
