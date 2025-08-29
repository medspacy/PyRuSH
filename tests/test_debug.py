import os
import sys
from loguru import logger
logger.remove()
logger.add(sys.stdout, level="DEBUG")
from spacy.lang.en import English
from PyRuSH.PyRuSHSentencizer import PyRuSHSentencizer

rule_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf", "rush_rules.tsv")
text_whitespace = "First sentence.    Second sentence before spaces.\nThird sentence after newline."


def test_whitespace_edge_split():    
    nlp = English()
    nlp.add_pipe("medspacy_pyrush", config={
        "rules_path": rule_path,
        "merge_gaps": False,
        "max_sentence_length": 20
    })
    sentencizer = nlp.get_pipe("medspacy_pyrush")
    doc = English()(text_whitespace)
    print("Tokens and indices:")
    for i, token in enumerate(doc):
        print(f"{i}: '{token.text}' idx={token.idx}")
    doc_guesses = sentencizer.predict([doc])[0]
    logger.info(f"doc_guesses: {doc_guesses}")
    serialized = str([(d, l) for d, l in zip(list(doc), doc_guesses)])
    logger.info(f"Serialized: {serialized}")
    goal = "[(First, True), (sentence, False), (., False), (   , True), (Second, True), (sentence, False), (before, True), (spaces, False), (., False), (\n, True), (Third, True), (sentence, False), (after, False), (newline, True), (., False)]"
    logger.info(f"Goal: {goal}")
    assert (serialized == goal)


def test_wrapped_split():
    nlp = English()
    nlp.add_pipe("medspacy_pyrush", config={
        "rules_path": rule_path,
        "merge_gaps": False,
        "max_sentence_length": 20
    })
    sentencizer = nlp.get_pipe("medspacy_pyrush")
    doc = nlp(text_whitespace)
    for sent in doc.sents:
        logger.info(f'{sent}---length:{len(sent.text)}')
        assert(len(sent.text) <= 20)


def test_wrapped_split_mergegap():
    nlp = English()
    nlp.add_pipe("medspacy_pyrush", config={
        "rules_path": rule_path,
        "merge_gaps": True,
        "max_sentence_length": 20
    })
    sentencizer = nlp.get_pipe("medspacy_pyrush")
    doc = nlp(text_whitespace)
    for sent in doc.sents:
        logger.info(f'{sent}---length:{len(sent.text)}')
        assert(len(sent.text) <= 20)
