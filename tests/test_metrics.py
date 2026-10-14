from ai_buddy.audio import SpeechAnalysis
from ai_buddy.metrics import Answer, compute_metrics, count_fillers, repeated_phrases, sentence_stats


def test_hesitations_counted():
    f = count_fillers("Um, so I was, uh, thinking. Umm yes. Er, hmm.")
    assert f["um"] == 2 and f["uh"] == 1 and f["er"] == 1 and f["hmm"] == 1


def test_like_as_filler_vs_verb_and_simile():
    assert count_fillers("I like pizza and they like pasta.")["like"] == 0
    assert count_fillers("It was like a dream. It looks like rain.")["like"] == 0
    assert count_fillers("So like I was thinking about the like big windows.")["like"] == 2
    assert count_fillers("It was, like, amazing. Like, really good.")["like"] == 2


def test_so_as_filler_vs_intensifier():
    assert count_fillers("I ate so much and it was so good, so that was fun.")["so"] == 0
    assert count_fillers("So, today I want to talk about sleep. Um so yeah.")["so"] == 2
    assert count_fillers("So today I want to talk about sleep.")["so"] == 1


def test_you_know_and_i_mean():
    f = count_fillers("It was, you know, fine. Do you know what time it is? I mean, it was okay. I mean it.")
    assert f["you know"] == 1
    assert f["I mean"] == 1


def test_kind_of_hedge_vs_type():
    assert count_fillers("What kind of job is it? I kind of liked it.")["kind of"] == 1


def test_unpunctuated_whisper_output():
    # FLM Whisper sometimes returns transcripts without punctuation. "you know the one" is
    # ambiguous without commas, so only the unambiguous fillers are asserted here.
    f = count_fillers("Um so like I was thinking that we should um maybe try the new place downtown "
                      "You know the one with the like big windows")
    assert f["um"] == 2 and f["like"] == 2 and f["so"] == 1


def test_sentence_stats_needs_punctuation():
    assert sentence_stats("one two three four five six seven eight nine ten " * 6) is None
    s = sentence_stats("This is short. " * 5 + "This one is a much longer sentence that keeps going on. ")
    assert s["count"] == 6 and s["longest_words"] == 11


def test_repeated_phrases():
    text = "the main point is clear. the main point is simple. so the main point is this."
    assert repeated_phrases(text) == [["the main point is", 3]]
    assert repeated_phrases("So today I want to talk about sleep. " * 4) == [["so today i want to talk about sleep", 4]]


def test_compute_metrics_presentation():
    text = "Um so today I want to talk about sleep. " * 10  # 9 words each -> 90 words
    speech = SpeechAnalysis(segments=[(0.5, 20.0), (23.0, 45.5)], duration_s=47.0)
    m = compute_metrics([Answer(text, speech)], long_pause_s=2.0, wpm_low=120, wpm_high=160)
    assert m["words"] == 90
    assert m["speaking_time_s"] == 45.0
    assert m["wpm"] == 120
    assert m["fillers"]["by_word"]["um"] == 10
    assert m["pauses"]["count"] == 1 and m["pauses"]["longest_s"] == 3.0
    assert m["pace"] == "good"
    assert "answers" not in m


def test_compute_metrics_interview_answers():
    a1 = Answer("I led a team of five engineers.", SpeechAnalysis([(0.0, 3.0)], 3.5), 0)
    a2 = Answer("Um, I think my biggest weakness is, like, delegating.", SpeechAnalysis([(0.2, 5.2)], 6), 10)
    m = compute_metrics([a1, a2])
    assert len(m["answers"]) == 2
    assert m["answers"][1]["fillers"] == 2
    assert m["longest_answer_s"] == 5.0
