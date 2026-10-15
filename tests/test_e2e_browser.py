"""Browser end-to-end tests: the real UI in headless Chrome, with a scripted microphone.

getUserMedia is replaced by a MediaStream the test controls, so speech (synthesized by Kokoro)
plays into the real capture worklet exactly while "Space" is held. Everything after the
microphone is the production path: resampling worklet, WebSocket, STT, LLM, TTS, playback, UI.

Run: uv run pytest tests/test_e2e_browser.py -m lemonade -s   (several minutes)
Screenshots land in test-artifacts/.
"""

from __future__ import annotations

import io
import time
import wave
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect, sync_playwright

from conftest import say, silence

pytestmark = pytest.mark.lemonade
ART = Path(__file__).parent.parent / "test-artifacts"

FAKE_MIC = """
(() => {
  const ctx = new AudioContext();
  const dest = ctx.createMediaStreamDestination();
  window.__fakeMic = {
    async play(url) {
      await ctx.resume();
      const data = await (await fetch(url)).arrayBuffer();
      const buf = await ctx.decodeAudioData(data);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(dest);
      src.start();
      return new Promise((resolve) => { src.onended = () => resolve(buf.duration); });
    },
  };
  navigator.mediaDevices.getUserMedia = async () => { await ctx.resume(); return dest.stream; };
})();
"""


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture(scope="module")
def page(server):
    ART.mkdir(exist_ok=True)
    clips = {
        "question": silence(0.2) + say("Hi Bella! I just finished a long bike ride along the coast. "
                                       "What should I eat to recover?") + silence(0.3),
        "todo": silence(0.2) + say("Please add buy bread to my to-do list.") + silence(0.3),
        "story": silence(0.2) + say("Can you tell me a really long story about a lighthouse keeper?") + silence(0.3),
        "talk": silence(0.6) + say(
            "Hello everyone. Today I want to talk about, um, why walking meetings are great. "
            "So, basically, walking gets your blood flowing and, like, helps people think more clearly.")
        + silence(2.6) + say("Second, uh, people are more honest when they walk side by side. "
                             "And finally, you know, it's just good for your health. Thanks for listening."),
        "small_talk": silence(0.2) + say("Hi Jordan, nice to meet you! I'm a software engineer, and um, "
                                         "I came here to learn about robotics. What brings you here?") + silence(0.3),
        "handsfree": silence(1.0) + say("What's a good name for a goldfish?") + silence(1.6),
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True,
                                    args=["--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 1180, "height": 860})
        ctx.add_init_script(FAKE_MIC)
        ctx.route("**/__fake/*.wav", lambda route: route.fulfill(
            body=wav_bytes(clips[route.request.url.rsplit("/", 1)[1][:-4]]), content_type="audio/wav"))
        pg = ctx.new_page()
        pg.on("console", lambda m: print(f"  [console.{m.type}] {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: print(f"  [pageerror] {e}"))
        pg.goto(server["base"])
        yield pg
        browser.close()


def wait_reply_done(page: Page, thread: str, timeout: int = 60_000) -> None:
    expect(page.locator(f"{thread} .msg.buddy .bubble.typing")).to_have_count(0, timeout=timeout)
    expect(page.locator("#dock")).to_have_attribute("data-busy", "false", timeout=timeout)
    expect(page.locator("#dock")).to_have_attribute("data-state", "idle", timeout=timeout)


def speak_holding_space(page: Page, clip: str) -> float:
    page.keyboard.down("Space")
    page.evaluate(f"window.__fakeMic.play('/__fake/{clip}.wav')")
    page.keyboard.up("Space")
    return time.monotonic()


def test_chat_push_to_talk(page: Page):
    page.locator(".tabs button[data-mode=chat]").click()
    expect(page.locator("#status-text")).to_have_text("Hold Space or the button to talk.", timeout=180_000)
    page.screenshot(path=ART / "01-chat-empty.png")
    speak_holding_space(page, "question")
    expect(page.locator(".msg.user .bubble").last).to_contain_text("bike", timeout=30_000)
    expect(page.locator("#dock")).to_have_attribute("data-state", "speaking", timeout=30_000)
    first_audio = page.evaluate("window.aiBuddyLatencyMs") / 1000
    wait_reply_done(page, "#chat-thread")
    print(f"\n  release -> first buddy audio (measured in page): {first_audio:.2f}s")
    print(f"  heard: {page.locator('.msg.user .bubble').last.inner_text()!r}")
    print(f"  reply: {page.locator('.msg.buddy .bubble').last.inner_text()!r}")
    page.screenshot(path=ART / "02-chat-reply.png")


def test_permission_card_for_an_mcp_tool(page: Page):
    """The to-do server's add_item changes something, so the assistant asks; the card's Yes button answers."""
    page.locator(".tabs button[data-mode=chat]").click()
    wait_reply_done(page, "#chat-thread")
    speak_holding_space(page, "todo")
    card = page.locator(".confirm").last
    expect(card).to_be_visible(timeout=150_000)
    expect(card.locator(".confirm-detail")).to_contain_text("todo_add_item")
    page.screenshot(path=ART / "02b-permission-card.png")
    card.get_by_role("button", name="Yes, go ahead").click()
    expect(card).to_have_class("confirm allowed", timeout=10_000)
    expect(card).to_contain_text("You said yes")
    wait_reply_done(page, "#chat-thread", timeout=150_000)


def test_scrolling_up_stays_put_while_a_reply_streams(page: Page):
    page.locator(".tabs button[data-mode=chat]").click()
    wait_reply_done(page, "#chat-thread", timeout=150_000)
    page.evaluate("""() => {
      for (let i = 0; i < 10; i++) {
        onEvent({type: "reply_start", id: 9000 + i});
        onEvent({type: "reply_delta", id: 9000 + i, text: "Filler line. ".repeat(40)});
        onEvent({type: "reply_end", id: 9000 + i, text: "", interrupted: false, timing: {}});
      }
      onEvent({type: "reply_start", id: 9999});
    }""")
    page.evaluate("onEvent({type: 'reply_delta', id: 9999, text: 'Streaming. '.repeat(40)})")
    assert page.evaluate("scroller.atBottom()")                 # followed while at the bottom
    page.mouse.move(500, 300)
    page.mouse.wheel(0, -700)
    page.wait_for_timeout(300)
    y = page.evaluate("window.scrollY")
    for _ in range(8):
        page.evaluate("onEvent({type: 'reply_delta', id: 9999, text: 'More words. '.repeat(30)})")
    assert page.evaluate("window.scrollY") == y                 # never pulled back down
    expect(page.locator("#jump-latest")).to_be_visible()
    page.locator("#jump-latest").click()
    assert page.evaluate("scroller.atBottom()")
    expect(page.locator("#jump-latest")).to_be_hidden()
    page.evaluate("onEvent({type: 'reply_end', id: 9999, text: '', interrupted: false, timing: {}})")


def test_settings_skills_mcp_and_voice_samples(page: Page):
    page.locator(".tabs button[data-mode=chat]").click()
    page.locator("#open-settings").click()
    mcp = page.locator("#set-mcp li.skill").filter(has_text="Todo")
    expect(mcp.locator(".badge")).to_contain_text("Connected · 3 tools", timeout=30_000)
    expect(mcp.locator(".tag.asks")).to_be_visible()
    page.screenshot(path=ART / "11-settings.png", full_page=True)
    # Practice tabs come from the speaking coach skill.
    coach = page.locator("#set-skills li.skill").filter(has_text="Speaking coach").locator("input")
    coach.click(force=True)
    expect(page.locator(".tabs button[data-mode=presentation]")).to_be_hidden(timeout=5_000)
    expect(page.locator(".tabs button[data-mode=interview]")).to_be_hidden()
    coach.click(force=True)
    expect(page.locator(".tabs button[data-mode=presentation]")).to_be_visible(timeout=5_000)
    # One voice sample at a time; picking another voice stops the first; Stop stops it.
    expect(page.locator("#stop-voice")).to_be_disabled()
    page.locator("#preview-voice").click()
    expect(page.locator("#stop-voice")).to_be_enabled()
    page.select_option("#set-voice", "bf_emma")
    expect(page.locator(".persona-name").first).to_have_text("Emma", timeout=10_000)
    page.wait_for_function("settingsUI.sample && settingsUI.sample.src.includes('bf_emma')", timeout=15_000)
    page.locator("#stop-voice").click()
    assert page.evaluate("settingsUI.sample") is None
    page.select_option("#set-voice", "af_bella")
    expect(page.locator(".persona-name").first).to_have_text("Bella", timeout=10_000)
    page.locator("#close-settings").click()
    assert page.evaluate("settingsUI.sample") is None          # closing Settings stops a sample


def test_barge_in_from_browser(page: Page):
    speak_holding_space(page, "story")
    expect(page.locator("#dock")).to_have_attribute("data-state", "speaking", timeout=30_000)
    page.wait_for_timeout(800)
    page.keyboard.down("Space")                     # interrupt the buddy mid-sentence
    expect(page.locator("#dock")).to_have_attribute("data-state", "listening", timeout=2_000)
    page.keyboard.up("Space")
    expect(page.locator(".msg.buddy .meta").last).to_have_text("interrupted", timeout=10_000)
    expect(page.locator(".toast").last).to_contain_text("didn't hear", timeout=10_000)


def test_presentation_flow(page: Page):
    page.locator(".tabs button[data-mode=presentation]").click()
    talk = page.locator("#talk")
    expect(talk).to_have_attribute("aria-label", "Start talk")
    page.screenshot(path=ART / "03-presentation-intro.png")
    talk.click()
    expect(page.locator("#pres-timer")).not_to_have_text("0:00", timeout=5_000)
    duration = page.evaluate("window.__fakeMic.play('/__fake/talk.wav')")
    page.screenshot(path=ART / "04-presentation-recording.png")
    t0 = time.monotonic()
    talk.click()                                    # Stop
    expect(page.locator(".tile").first).to_be_visible(timeout=60_000)
    t_metrics = time.monotonic() - t0
    expect(page.locator("#pres-results .feedback h2").first).to_be_visible(timeout=240_000)
    expect(page.get_by_role("button", name="New talk")).to_be_enabled(timeout=240_000)
    print(f"\n  talk {duration:.0f}s; metrics after {t_metrics:.1f}s; feedback complete after {time.monotonic() - t0:.0f}s")
    tiles = page.locator(".tile").all_inner_texts()
    print("  tiles:", [t.replace("\n", " ") for t in tiles])
    page.screenshot(path=ART / "05-presentation-feedback.png", full_page=True)
    assert "1" in tiles[2]                          # one long pause (2.6 s + clip padding)
    assert page.locator(".chip").count() >= 3       # um, so, like, uh, you know...


def test_conversation_practice_flow(page: Page):
    page.locator(".tabs button[data-mode=interview]").click()
    page.locator(".scenario[data-key=small_talk]").click()
    page.fill("#iv-details", "a robotics meetup")
    page.screenshot(path=ART / "06-practice-setup.png")
    page.get_by_role("button", name="Start").click()
    opener = page.locator("#iv-thread .msg.buddy .bubble").first
    expect(opener).not_to_have_text("", timeout=60_000)
    expect(page.locator("#dock")).to_have_attribute("data-state", "speaking", timeout=30_000)
    expect(page.locator("#dock")).to_have_attribute("data-state", "idle", timeout=60_000)
    print(f"\n  stranger: {opener.inner_text()!r}")
    speak_holding_space(page, "small_talk")
    expect(page.locator("#iv-thread .msg.user .bubble").last).to_contain_text("robotics", timeout=30_000)
    expect(page.locator("#iv-thread .msg.buddy")).to_have_count(2, timeout=30_000)
    wait_reply_done(page, "#iv-thread")
    reply = page.locator("#iv-thread .msg.buddy .bubble").last.inner_text()
    print(f"  stranger: {reply!r}")
    assert reply.strip()
    page.screenshot(path=ART / "07-practice-conversation.png")
    t0 = time.monotonic()
    page.get_by_role("button", name="End & get feedback").click()
    expect(page.locator("#iv-results .feedback h2").first).to_be_visible(timeout=240_000)
    expect(page.get_by_role("button", name="Practice again")).to_be_enabled(timeout=240_000)
    print(f"  feedback complete {time.monotonic() - t0:.0f}s after End (includes the 35B swap)")
    page.screenshot(path=ART / "08-practice-feedback.png", full_page=True)


def test_history_view(page: Page):
    page.locator(".tabs button[data-mode=history]").click()
    expect(page.locator(".trend svg")).to_have_count(3, timeout=10_000)
    rows = page.locator("table.sessions tbody tr")
    assert rows.count() >= 3
    box = page.locator(".trend svg").first.bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.5)
    expect(page.locator(".viz-tip")).to_be_visible()
    page.screenshot(path=ART / "09-history.png", full_page=True)
    page.emulate_media(color_scheme="dark")
    page.mouse.move(box["x"] + box["width"] * 0.9, box["y"] + box["height"] * 0.5)
    page.screenshot(path=ART / "09b-history-dark.png", full_page=True)
    page.emulate_media(color_scheme="light")
    rows.first.click()
    expect(page.locator("dialog.detail")).to_be_visible()
    page.screenshot(path=ART / "10-history-detail.png")
    page.get_by_role("button", name="Close").click()


def test_hands_free(page: Page):
    page.locator(".tabs button[data-mode=chat]").click()
    expect(page.locator("#status-text")).to_contain_text("talk", timeout=60_000)
    page.locator("#handsfree").check()
    expect(page.locator("#status-text")).to_have_text("Hands-free: just start talking.", timeout=60_000)
    n_user = page.locator("#chat-thread .msg.user").count()
    page.evaluate("window.__fakeMic.play('/__fake/handsfree.wav')")
    expect(page.locator("#chat-thread .msg.user")).to_have_count(n_user + 1, timeout=20_000)
    expect(page.locator("#chat-thread .msg.user .bubble").last).to_contain_text("goldfish", timeout=30_000)
    expect(page.locator("#dock")).to_have_attribute("data-state", "speaking", timeout=30_000)
    print(f"\n  hands-free heard: {page.locator('#chat-thread .msg.user .bubble').last.inner_text()!r}")
    page.locator("#handsfree").uncheck()


def test_silent_talk_reenables_start(page: Page):
    page.locator(".tabs button[data-mode=presentation]").click()
    if page.get_by_role("button", name="New talk").count():
        page.get_by_role("button", name="New talk").click()
    talk = page.locator("#talk")
    expect(talk).to_be_enabled(timeout=60_000)
    talk.click()                                   # Start, then say nothing
    page.wait_for_timeout(2500)
    talk.click()                                   # Stop
    expect(page.locator(".toast").last).to_contain_text("didn't catch any speech", timeout=60_000)
    expect(talk).to_be_enabled(timeout=5_000)
    expect(talk).to_have_attribute("aria-label", "Start talk")
    expect(page.locator("#pres-intro")).to_be_visible()
