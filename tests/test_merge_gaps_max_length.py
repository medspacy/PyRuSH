import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

import pytest
import spacy
from PyRuSH.StaticSentencizerFun import cpredict_merge_gaps

def dummy_sentencizer(text):
    # Dummy sentencizer: splits on periods and newlines
    spans = []
    start = 0
    split=False
    for i, c in enumerate(text):
        if split:
            spans.append(type('Span', (), {'begin': start, 'end': i+1})())
            start = i+1
            split=False
        if c in '.\n':
            split=True            
    if start < len(text):
        spans.append(type('Span', (), {'begin': start, 'end': len(text)})())
    return spans


def dummy_sentencizer2(text):
    # Dummy sentencizer: splits on periods and newlines
    spans = []
    start = 0
    for i, c in enumerate(text):
        if c in '.\n':
            spans.append(type('Span', (), {'begin': start, 'end': i+1})())
            start = i+1
    if start < len(text):
        spans.append(type('Span', (), {'begin': start, 'end': len(text)})())
    return spans

def test_merge_gaps_basic():
    nlp = spacy.blank('en')
    doc = nlp("This is a sentence. This is another one.")
    spans = dummy_sentencizer(doc.text)
    print("dummy_sentencizer spans:", [(span.begin, span.end, doc.text[span.begin:span.end]) for span in spans])
    print("Tokens:")
    for i, token in enumerate(doc):
        print(f"  idx={i}, text='{token.text}', token.idx={token.idx}")
    guesses = cpredict_merge_gaps([doc], dummy_sentencizer)
    print("cpredict_merge_gaps sentence starts:", [(i, token.text) for i, token in enumerate(doc) if guesses[0][i]])
    print("guesses:", guesses[0])
    assert guesses[0].count(True) == 2
    # Verify split sentence character length is non-zero
    starts = [i for i, v in enumerate(guesses[0]) if v]
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(doc)
        sentence_text = "".join([doc[i].text_with_ws for i in range(start, end)])
        char_len = len(sentence_text)
        assert char_len > 0

def test_merge_gaps_basic2():
    nlp = spacy.blank('en')
    doc = nlp("This is a sentence. This is another one.")
    spans = dummy_sentencizer2(doc.text)
    print("dummy_sentencizer spans:", [(span.begin, span.end, doc.text[span.begin:span.end]) for span in spans])
    print("Tokens:")
    for i, token in enumerate(doc):
        print(f"  idx={i}, text='{token.text}', token.idx={token.idx}")
    guesses = cpredict_merge_gaps([doc], dummy_sentencizer2)
    print("cpredict_merge_gaps sentence starts:", [(i, token.text) for i, token in enumerate(doc) if guesses[0][i]])
    print("guesses:", guesses[0])
    assert guesses[0].count(True) == 2
    starts = [i for i, v in enumerate(guesses[0]) if v]
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(doc)
        sentence_text = "".join([doc[i].text_with_ws for i in range(start, end)])
        char_len = len(sentence_text)
        assert char_len > 0


def test_merge_gaps_max_length():
    nlp = spacy.blank('en')
    doc = nlp("A very long sentence that should be split at whitespace before the max length is reached.")
    max_len = 20
    spans = dummy_sentencizer(doc.text)
    print("dummy_sentencizer spans:", [(span.begin, span.end, doc.text[span.begin:span.end]) for span in spans])
    guesses = cpredict_merge_gaps([doc], dummy_sentencizer, max_sentence_length=max_len)
    print("cpredict_merge_gaps sentence starts:", [(i, token.text) for i, token in enumerate(doc) if guesses[0][i]])
    # Should split at least once
    assert guesses[0].count(True) > 1
    starts = [i for i, v in enumerate(guesses[0]) if v]
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(doc)
        sentence_text = "".join([doc[i].text_with_ws for i in range(start, end)])
        char_len = len(sentence_text)
        assert char_len <= max_len, f"Sentence from {start} to {end} has char length {char_len} > {max_len}"

def test_merge_gaps_whitespace_edge():
    nlp = spacy.blank('en')
    doc = nlp("First sentence.    Second sentence after spaces.\nThird sentence after newline.")
    spans = dummy_sentencizer(doc.text)
    print("dummy_sentencizer spans:", [(span.begin, span.end, doc.text[span.begin:span.end]) for span in spans])
    guesses = cpredict_merge_gaps([doc], dummy_sentencizer, max_sentence_length=15)
    print("cpredict_merge_gaps sentence starts:", [(i, token.text) for i, token in enumerate(doc) if guesses[0][i]])
    # Should split at whitespace/newline before max length
    assert guesses[0].count(True) >= 3
    starts = [i for i, v in enumerate(guesses[0]) if v]
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(doc)
        sentence_text = "".join([doc[i].text_with_ws for i in range(start, end)])
        char_len = len(sentence_text)
        assert char_len <= 15, f"Sentence from {start} to {end} has char length {char_len} > 15"
