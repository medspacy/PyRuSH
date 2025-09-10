import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyRuSH import PyRuSHSentencizer
from spacy.lang.en import English


class TestRuSH(unittest.TestCase):

    def setUp(self):
        self.pwd = os.path.dirname(os.path.abspath(__file__))

    # def test_doc(self):
    #     nlp = English()
    #     nlp.add_pipe("medspacy_pyrush")
    #     doc = nlp("This is a sentence. This is another sentence.")
    #     print('\n'.join([str(s) for s in doc.sents]))
    #     print('\nTotal sentences: {}'.format(len([s for s in doc.sents])))
    #     print('\ndoc is an instance of {}'.format(type(doc)))

    def test_doc4(self):
        input_str='''Ms. [**Known patient lastname 2004**] was admitted on [**2573-5-30**]. Ultrasound
at the time of admission demonstrated pancreatic duct dilitation and
edematous gallbladder. She was admitted to the ICU.
Discharge Medications:
1. Miconazole Nitrate 2 % Powder Sig: One (1) Appl Topical  BID
(2 times a day) as needed.
2. Heparin Sodium (Porcine) 5,000 unit/mL Solution Sig: One (1)
Injection TID (3 times a day).
3. Acetaminophen 160 mg/5 mL Elixir Sig: One (1)  PO Q4-6H
(every 4 to 6 hours) as needed.'''
        nlp = English()
        nlp.add_pipe("medspacy_pyrush", config={"rules_path": os.path.join(self.pwd, 'rush_rules.tsv')})
        nlp.initialize()
        doc = nlp(input_str)
        sents = [s for s in doc.sents]
        for sent in sents:
            print('>' + str(sent) + '<\n\n')
        assert(sents[-1].text=='''Sig: One (1)  PO Q4-6H
(every 4 to 6 hours) as needed.''')

if __name__ == '__main__':
    unittest.main()
