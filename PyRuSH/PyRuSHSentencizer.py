# ******************************************************************************
#  MIT License
#
#  Copyright (c) 2020 Jianlin Shi
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
#  files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
#  modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ******************************************************************************
from spacy import Language
from spacy.pipeline import Sentencizer

from .RuSH import RuSH
from .StaticSentencizerFun import cpredict_merge_gaps,cpredict_split_gaps, cset_annotations


@Language.factory("medspacy_pyrush")
class PyRuSHSentencizer(Sentencizer):
    def __init__(self, nlp: Language, name: str = "medspacy_pyrush", rules_path: str = '', max_repeat: int = 50,
                 auto_fix_gaps: bool = True, merge_gaps: bool = False, max_sentence_length: int = None) -> Sentencizer:
        """
        Initialize the PyRuSH sentencizer component.

        Args:
            nlp (Language): The spaCy language pipeline.
            name (str): Name of the component. Default is "medspacy_pyrush".
            rules_path (str): Path to the rule file or rules themselves. If empty, defaults to 'conf/rush_rules.tsv'.
            max_repeat (int): Maximum number of repeats allowed for the '+' wildcard in rules.
            auto_fix_gaps (bool): If True, attempts to fix gaps caused by malformed rules.
            merge_gaps (bool): If True, merges gaps between sentences into the preceding sentence. If False, splits gaps (might be multiple whitespaces or new line characters) into separate sentences.
            max_sentence_length (int or None): Maximum allowed sentence length in characters. If set, sentences longer than this will be split.

        Notes:
            - Setting merge_gaps controls whether gaps are merged or split.
            - max_sentence_length applies to both merge and split modes.
        """
        self.nlp = nlp
        self.name = name
        if rules_path is None or rules_path == '':
            import os
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            rules_path = str(os.path.join(root, 'conf', 'rush_rules.tsv'))
        self.rules_path = rules_path
        self.rush = RuSH(rules=rules_path, max_repeat=max_repeat, auto_fix_gaps=auto_fix_gaps)
        self.merge_gaps = merge_gaps
        self.max_sentence_length = max_sentence_length

    @classmethod
    def from_nlp(cls, nlp, **cfg):
        """
        Create a PyRuSHSentencizer instance from a spaCy nlp object and configuration.

        Args:
            nlp (Language): The spaCy language pipeline.
            **cfg: Additional configuration parameters for initialization.

        Returns:
            PyRuSHSentencizer: An initialized sentencizer instance.
        """
        return cls(**cfg)

    def __call__(self, doc):
        """
        Apply sentence boundary detection to a spaCy Doc and set sentence start annotations.

        Args:
            doc (Doc): The spaCy Doc to process.

        Returns:
            Doc: The processed Doc with sentence boundaries set.
        """
        tags = self.predict([doc])
        cset_annotations([doc], tags)
        return doc

    def predict(self, docs):
        """
        Predict sentence boundaries for a batch of spaCy Docs.

        Args:
            docs (list of Doc): List of spaCy Docs to process.

        Returns:
            list of list of bool: Sentence start guesses for each Doc.

        Notes:
            - Does not modify the Docs; only returns sentence start predictions.
        """
        if self.merge_gaps:
            guesses = cpredict_merge_gaps(docs, self.rush.segToSentenceSpans, self.max_sentence_length)
        else:
            guesses = cpredict_split_gaps(docs, self.rush.segToSentenceSpans, self.max_sentence_length)
        return guesses

    def set_annotations(self, docs, batch_tag_ids, tensors=None):
        """
        Set sentence boundary annotations on spaCy Docs.

        Args:
            docs (list of Doc): List of spaCy Docs to annotate.
            batch_tag_ids (list of list of bool): Sentence start tags for each Doc.
            tensors: Placeholder for future extensions (optional).

        Notes:
            - This method overwrites spaCy's Sentencizer annotations.
        """
        cset_annotations(docs, batch_tag_ids, tensors)
