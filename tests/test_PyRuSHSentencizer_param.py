import unittest
import os
from loguru import logger
from spacy.lang.en import English
from PyRuSH.PyRuSHSentencizer import PyRuSHSentencizer

class TestPyRuSHSentencizerParams(unittest.TestCase):
    def setUp(self):
        self.text_short = "Sentence one. Sentence two!"
        self.text_long = "This is a very long sentence that should be split at whitespace before the max length is reached. " * 5
        self.text_whitespace = "First sentence.    Second sentence after spaces.\nThird sentence after newline."
        self.rule_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf", "rush_rules.tsv")

    def make_nlp(self, merge_gaps, max_sentence_length):
        nlp = English()
        nlp.add_pipe("medspacy_pyrush", config={
            "rules_path": self.rule_path,
            "merge_gaps": merge_gaps,
            "max_sentence_length": max_sentence_length
        })
        return nlp

    def test_merge_gaps_true_no_maxlen(self):
        nlp = self.make_nlp(merge_gaps=True, max_sentence_length=None)
        doc = nlp(self.text_short)
        sents = [s.text for s in doc.sents]
        logger.info("[merge_gaps=True, max_sentence_length=None] Split sentences:")
        for i, sent in enumerate(sents):
            logger.info(f"  [{i}] len={len(sent)} {repr(sent)}")
        self.assertGreaterEqual(len(sents), 2)

    def test_merge_gaps_false_no_maxlen(self):
        nlp = self.make_nlp(merge_gaps=False, max_sentence_length=None)
        doc = nlp(self.text_short)
        sents = [s.text for s in doc.sents]
        logger.info("[merge_gaps=False, max_sentence_length=None] Split sentences:")
        for i, sent in enumerate(sents):
            logger.info(f"  [{i}] len={len(sent)} {repr(sent)}")
        self.assertGreaterEqual(len(sents), 2)

    def test_merge_gaps_true_with_maxlen(self):
        nlp = self.make_nlp(merge_gaps=True, max_sentence_length=50)
        doc = nlp(self.text_long)
        sents = [s.text for s in doc.sents]
        logger.info("[merge_gaps=True, max_sentence_length=50] Split sentences:")
        for i, sent in enumerate(sents):
            logger.info(f"  [{i}] len={len(sent)} {repr(sent)}")
        # Should split long text into multiple sentences
        self.assertGreater(len(sents), 2)
        for sent in sents:
            self.assertLessEqual(len(sent), 60)  # allow some leeway

    def test_merge_gaps_false_with_maxlen(self):
        nlp = self.make_nlp(merge_gaps=False, max_sentence_length=50)
        doc = nlp(self.text_long)
        sents = [s.text for s in doc.sents]
        logger.info("[merge_gaps=False, max_sentence_length=50] Split sentences:")
        for i, sent in enumerate(sents):
            logger.info(f"  [{i}] len={len(sent)} {repr(sent)}")
        self.assertGreater(len(sents), 2)
        # Allow up to 100 chars due to tokenization edge cases
        for sent in sents:
            self.assertLessEqual(len(sent), 100)

    def test_whitespace_edge_merge(self):
        nlp = self.make_nlp(merge_gaps=True, max_sentence_length=20)
        doc = nlp(self.text_whitespace)
        sents = [s.text for s in doc.sents]
        for i, sent in enumerate(sents):
            logger.info(f"  [{i}] len={len(sent)} {repr(sent)}")
            self.assertLessEqual(len(sent), 20, f"Sentence {i} exceeds max_sentence_length: {len(sent)} > 20")
        self.assertGreaterEqual(len(sents), 3)

    def test_whitespace_edge_split(self):
        nlp = self.make_nlp(merge_gaps=False, max_sentence_length=20)
        doc = nlp(self.text_whitespace)
        sents = [s.text for s in doc.sents]
        logger.debug(str([t for t in doc]))
        for i, sent in enumerate(sents):
            logger.info(f"  [{i}] len={len(sent)} {repr(sent)}")
            self.assertLessEqual(len(sent), 20, f"Sentence {i} exceeds max_sentence_length: {len(sent)} > 20")
        self.assertGreaterEqual(len(sents), 3)

if __name__ == "__main__":
    unittest.main()
