import pytest
from PyRuSH.StaticSentencizerFun import cpredict_split_gaps
import spacy
from loguru import logger
nlp = spacy.blank("en")


def dummy_sentencizer_fun(text):
    # For testing, split sentences at every period
    spans = []
    start = 0
    for i, c in enumerate(text):
        if c == ".":
            spans.append((start, i+1))
            start = i+1
    if start < len(text):
        spans.append((start, len(text)))
    return spans

def make_doc_from_text(text):
    # Use spaCy's default tokenizer
    return nlp(text)

def test_split_gaps_single_token():
    doc = make_doc_from_text("Hello")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    assert starts == [0]

def test_split_gaps_single_period():
    doc = make_doc_from_text(".")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    assert starts == [0]

def test_split_gaps_consecutive_periods():
    doc = make_doc_from_text("Hello..World.")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    # Should mark the first token and after each period
    assert starts[0] == 0
    assert len(starts) >= 2

def test_split_gaps_long_sentence_no_period():
    doc = make_doc_from_text("A " * 100)
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun, 20)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    # Should split every ~10 tokens (since each token is 1 char + 1 space)
    assert starts[0] == 0
    assert len(starts) > 1

def test_split_gaps_non_ascii():
    doc = make_doc_from_text("Hello 世界 . World .")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    # Get sentences by splitting at sentence start indices
    sentences = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(doc)
        sentences.append(" ".join([doc[i].text for i in range(start, end)]))
    logger.info(f"[test_split_gaps_non_ascii] Split sentences: {sentences}")
    # Expect sentences to be 'Hello 世界 .' and 'World .'
    assert any("世界" in s for s in sentences)
    assert any("World" in s for s in sentences)

def test_split_gaps_punctuation_only():
    doc = make_doc_from_text("!!! . ??? .")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    assert starts[0] == 0
    assert len(starts) > 1

def test_split_gaps_basic():
    doc = make_doc_from_text("This is a sentence. This is another one.")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    sentences = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(doc)
        sentences.append(" ".join([doc[i].text for i in range(start, end)]))
    logger.info(f"[test_split_gaps_basic] Split sentences: {sentences}")
    assert "This is a sentence ." in sentences
    assert "This is another one ." in sentences

def test_split_gaps_max_length_none():
    doc = make_doc_from_text("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z.")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun, None)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    assert starts == [0]

def test_split_gaps_max_length_set():
    doc = make_doc_from_text("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z.")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun, 10)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    assert starts[0] == 0
    assert len(starts) > 1

def test_split_gaps_empty_doc():
    doc = make_doc_from_text("")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun)
    assert guesses == [[]]

def test_split_gaps_whitespace_none():
    doc = make_doc_from_text("   .   .")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun, None)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    sentences = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(doc)
        sentences.append(" ".join([doc[i].text for i in range(start, end)]))
    sentences = [s.strip() for s in sentences]
    logger.info(f"[test_split_gaps_whitespace_none] Split sentences: {sentences}")
    # Should have two sentences, each with a single period
    assert sentences == [".", "."]

def test_split_gaps_whitespace_set():
    doc = make_doc_from_text("   .   .")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun, 5)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    sentences = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(doc)
        sentences.append(" ".join([doc[i].text for i in range(start, end)]))
    sentences = [s.strip() for s in sentences]
    logger.info(f"[test_split_gaps_whitespace_set] Split sentences: {sentences}")
    assert sentences == [".", "."]

def test_split_gaps_mixed_whitespace_and_text():
    doc = make_doc_from_text("   . Hello .   . World .")
    guesses = cpredict_split_gaps([doc], dummy_sentencizer_fun)
    starts = [i for i, v in enumerate(guesses[0]) if v]
    sentences = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(doc)
        sentences.append(" ".join([doc[i].text for i in range(start, end)]))
    sentences = [s.strip() for s in sentences]
    logger.info(f"[test_split_gaps_mixed_whitespace_and_text] Split sentences: {sentences}")
    # Should have sentences: '.', 'Hello .', 'World .'
    assert "." in sentences
    assert "Hello ." in sentences
    assert "World ." in sentences