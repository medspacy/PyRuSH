import sys,os
from loguru import logger
logger.remove()
logger.add(sys.stdout, level="DEBUG")

def test_whitespace_edge_split():
    from spacy.lang.en import English
    from loguru import logger
    import medspacy
    text_whitespace = "First sentence.   Second sentence before spaces.\nThird sentence after newline."
    nlp = English()
    rule_path=os.path.join(os.path.dirname(__file__), 'rush_rules.tsv')
    nlp.add_pipe("medspacy_pyrush", config={
        "rules_path": rule_path,
        "merge_gaps": False,
        "max_sentence_length": 20
    })
    sentencizer = nlp.get_pipe("medspacy_pyrush")
    doc = nlp(text_whitespace)
    # Try to get the actual span function from RuSH
    spans=sentencizer.rush.segToSentenceSpans(text_whitespace)
    logger.debug('Print rush segmented spans: \n----------------\n')
    logger.debug(f"Spans: {[(span.begin, span.end) for span in spans]}\n----------------\n")
    logger.debug(f'Print token offsets: ')
    logger.debug(f'{[(t, t.idx) for t in doc]}')
    doc_guesses = sentencizer.predict([doc])[0]
    logger.debug(f"doc_guesses: {doc_guesses}")
    serialized = [(str(d), l) for d, l in zip(list(doc), doc_guesses)]
    logger.debug(f"Serialized: {serialized}")
    # Adjusted expected output to match spacy tokenization
    goal = [("First", True), ("sentence", False), (".", False), ("  ", True), ("Second", True), ("sentence", False), ("before", True), ("spaces", False), (".", False), ("\n", True), ("Third", True), ("sentence", False), ("after", False), ("newline", True), (".", False)]
    logger.debug(f"Goal: {goal}")
    for s, g in zip(serialized, goal):
        logger.debug(f'{s} == {g}' if s == g else f'{s} != {g}')
        assert s == g
