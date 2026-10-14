from ai_buddy.text import SpeechChunker, clean_for_speech, strip_emoji


def stream(chunker, text, step=3):
    out = []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i:i + step])
    return out + chunker.flush()


def test_first_chunk_cuts_early_at_clause():
    chunks = stream(SpeechChunker(), "Oh wow, a river walk sounds lovely. You should make pasta tonight!")
    assert chunks[0] == "Oh wow, a river walk sounds lovely."
    assert chunks[-1] == "You should make pasta tonight!"


def test_first_chunk_clause_needs_min_words():
    chunks = stream(SpeechChunker(), "That is a really great idea, honestly, and I think you will love it.")
    assert chunks[0] == "That is a really great idea,"
    assert "".join(chunks).replace(" ", "") == "That is a really great idea, honestly, and I think you will love it.".replace(" ", "")


def test_short_opener_is_spoken_first():
    chunks = stream(SpeechChunker(), "That sounds amazing—coastal rides are pure magic! Try a smoothie.")
    assert chunks[:2] == ["That sounds amazing—", "coastal rides are pure magic!"]


def test_unspaced_em_dash_is_a_clause_break():
    chunks = stream(SpeechChunker(), "That sounds really amazing—coastal rides are so scenic! Eat some pasta.")
    assert chunks[0] == "That sounds really amazing—"
    assert chunks[1] == "coastal rides are so scenic!"


def test_abbreviations_do_not_split():
    chunks = stream(SpeechChunker(), "I asked Dr. Smith about it. She said yes.")
    assert chunks == ["I asked Dr. Smith about it.", "She said yes."]


def test_long_sentence_splits_at_clause():
    long = ("I think the best way to prepare is to research the company, practice your stories out loud, "
            "and get a good night of sleep before the big day so you feel fresh.")
    chunks = stream(SpeechChunker(max_words=12), long)
    assert len(chunks) >= 2
    assert " ".join(chunks).split() == long.split()


def test_flush_returns_unterminated_tail():
    c = SpeechChunker()
    assert c.feed("Sounds good to me") == []
    assert c.flush() == ["Sounds good to me"]


def test_clean_for_speech_removes_emoji_markdown_and_stage_directions():
    assert clean_for_speech("**Great** job! 😊 *smiles* Let's go 🌄") == "Great job! Let's go"
    assert clean_for_speech("- point one — and more") == "point one, and more"
    assert clean_for_speech("🙂🙂") == ""
    assert clean_for_speech("That is *really* cool (laughs) and __fun__.") == "That is really cool and fun."


def test_strip_emoji():
    assert strip_emoji("Hi there! 😊 How are you? 🌲😄") == "Hi there! How are you?"
