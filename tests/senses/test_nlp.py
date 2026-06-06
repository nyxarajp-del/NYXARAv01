"""Tests for nyxara.senses.nlp."""

from __future__ import annotations

from nyxara.senses.nlp import (Analysis, Entity, NLP, SentimentResult, detect_language,
                               entities, intent, keyphrases, readability, sentences,
                               sentiment, summarize, tokenize)


# -------------------- tokenisation -------------------- #
def test_tokenize_words():
    assert tokenize("Hello, world!") == ["Hello", "world"]


def test_tokenize_lower():
    assert tokenize("Hello World", lower=True) == ["hello", "world"]


def test_tokenize_keep_punct():
    toks = tokenize("Hi, there!", keep_punct=True)
    assert "," in toks and "!" in toks


def test_tokenize_empty():
    assert tokenize("") == []


# -------------------- sentence splitting -------------------- #
def test_sentences_basic():
    assert sentences("Hello there. How are you? I am fine.") == \
        ["Hello there.", "How are you?", "I am fine."]


def test_sentences_abbreviation_not_split():
    out = sentences("Dr. Smith arrived. He was late.")
    assert out == ["Dr. Smith arrived.", "He was late."]


def test_sentences_empty():
    assert sentences("   ") == []


def test_sentences_single():
    assert sentences("Just one sentence") == ["Just one sentence"]


# -------------------- language detection -------------------- #
def test_detect_english():
    assert detect_language("the cat is in the house and you know it") == "en"


def test_detect_spanish():
    assert detect_language("el gato que esta en la casa de un amigo") == "es"


def test_detect_french():
    assert detect_language("le chat est dans la maison et un ami") == "fr"


def test_detect_german():
    assert detect_language("der hund ist in dem haus und von zu") == "de"


def test_detect_empty_unknown():
    assert detect_language("") == "unknown"


def test_detect_default_en():
    assert detect_language("xyzzy plugh") == "en"


# -------------------- entities -------------------- #
def test_entity_email():
    ents = entities("reach me at jp@example.com please")
    assert any(e.label == "EMAIL" and e.text == "jp@example.com" for e in ents)


def test_entity_url():
    ents = entities("see https://nyxara.ai/docs for more")
    assert any(e.label == "URL" for e in ents)


def test_entity_mention_hashtag():
    ents = entities("ping @jp about #security")
    labels = {e.label for e in ents}
    assert "MENTION" in labels and "HASHTAG" in labels


def test_entity_money_and_date():
    ents = entities("pay $50 on 2026-06-06")
    labels = {e.label for e in ents}
    assert "MONEY" in labels and "DATE" in labels


def test_entity_time():
    ents = entities("the meeting is at 14:30 today")
    assert any(e.label == "TIME" for e in ents)


def test_entity_proper_noun_multiword():
    ents = entities("a report about Acme Corp was filed")
    assert any(e.text == "Acme Corp" and e.label == "PROPER_NOUN" for e in ents)


def test_entity_sentence_start_single_cap_skipped():
    # a lone capitalised word starting a sentence is not a proper noun
    ents = entities("The server crashed.")
    assert not any(e.text == "The" for e in ents)


def test_entity_to_dict():
    e = Entity("x@y.com", "EMAIL", 0, 7)
    assert e.to_dict()["label"] == "EMAIL"


# -------------------- keyphrases -------------------- #
def test_keyphrases_surface_content():
    text = "Machine learning models require large training datasets and careful tuning."
    phrases = [p for p, _ in keyphrases(text)]
    assert any("learning" in p or "training" in p or "datasets" in p for p in phrases)


def test_keyphrases_respects_top():
    text = "alpha beta. gamma delta. epsilon zeta. eta theta. iota kappa."
    assert len(keyphrases(text, top=2)) <= 2


def test_keyphrases_empty():
    assert keyphrases("the and is of") == []  # all stopwords


# -------------------- sentiment -------------------- #
def test_sentiment_positive():
    r = sentiment("this is great and wonderful")
    assert r.label == "positive" and r.polarity > 0


def test_sentiment_negative():
    r = sentiment("this is terrible and awful")
    assert r.label == "negative" and r.polarity < 0


def test_sentiment_neutral():
    r = sentiment("the box is on the table")
    assert r.label == "neutral"


def test_sentiment_negation_flips():
    assert sentiment("this is good").polarity > 0
    assert sentiment("this is not good").polarity < 0


def test_sentiment_intensifier_amplifies():
    plain = sentiment("good").polarity
    strong = sentiment("very good").polarity
    assert strong >= plain  # intensifier does not reduce positivity


def test_sentiment_counts():
    r = sentiment("great great bad")
    assert r.positive == 2 and r.negative == 1


def test_sentiment_result_to_dict():
    assert SentimentResult(0.5, "positive", 1, 0).to_dict()["label"] == "positive"


# -------------------- intent -------------------- #
def test_intent_question_qmark():
    assert intent("are you ready?").kind == "question"


def test_intent_question_wh():
    assert intent("what is happening").kind == "question"


def test_intent_request_please():
    assert intent("please scan the network").kind == "request"


def test_intent_request_could_you():
    assert intent("could you check the logs").kind == "request"


def test_intent_command():
    assert intent("delete the old files").kind == "command"


def test_intent_greeting():
    assert intent("hello NYXARA").kind == "greeting"


def test_intent_gratitude():
    assert intent("thank you so much").kind == "gratitude"


def test_intent_affirmation_negation():
    assert intent("yes do it").kind == "affirmation"
    assert intent("no stop").kind == "negation"


def test_intent_statement():
    assert intent("the server is running fine").kind == "statement"


def test_intent_empty():
    assert intent("   ").kind == "empty"


def test_intent_has_cues():
    r = intent("why did it fail?")
    assert r.cues and r.confidence > 0


# -------------------- summarize -------------------- #
def test_summarize_short_text_returns_all():
    text = "One sentence. Two sentence."
    assert summarize(text, n=3) == sentences(text)


def test_summarize_picks_n_in_order():
    text = ("The network is critical. Cats are fluffy. The network firewall protects "
            "the network from attacks. Random trivia here. Network security matters most.")
    out = summarize(text, n=2)
    assert len(out) == 2
    # ordering preserved relative to source
    idx = [sentences(text).index(s) for s in out]
    assert idx == sorted(idx)


def test_summarize_empty():
    assert summarize("") == []


# -------------------- readability -------------------- #
def test_readability_counts():
    r = readability("The cat sat. The dog ran.")
    assert r["words"] == 6 and r["sentences"] == 2


def test_readability_flesch_range():
    r = readability("This is a simple short sentence.")
    assert 0.0 <= r["flesch_reading_ease"] <= 100.0


def test_readability_empty():
    r = readability("")
    assert r["words"] == 0 and r["flesch_reading_ease"] == 0.0


# -------------------- facade -------------------- #
def test_nlp_analyze_returns_analysis():
    a = NLP().analyze("Hello world. Please help me at jp@example.com.")
    assert isinstance(a, Analysis)
    assert a.language == "en"
    assert any(e.label == "EMAIL" for e in a.entities)
    assert a.intent.kind in {"greeting", "request", "statement", "command"}


def test_nlp_analyze_to_dict():
    d = NLP().analyze("The system works.").to_dict()
    assert "sentiment" in d and "intent" in d and "readability" in d


def test_nlp_staticmethod_passthroughs():
    assert NLP.tokenize("hi there") == ["hi", "there"]
    assert NLP.detect_language("the and is") == "en"
