# Copyright  2018  Department of Biomedical Informatics, University of Utah
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import os

from PyRuSH import RuSH

class TestRuSH(unittest.TestCase):

    def setUp(self):
        self.pwd = os.path.dirname(os.path.abspath(__file__))
        self.rush = RuSH(str(os.path.join(self.pwd, 'rush_rules.tsv')), enable_logger=True)

    def test1(self):
        input_str = 'Can Mr. K check it. Look\n good.\n'
        sentences = self.rush.segToSentenceSpans(input_str)
        assert (sentences[0].begin == 0 and sentences[0].end == 19)
        assert (sentences[1].begin == 20 and sentences[1].end == 31)

    def test2(self):
        input_str = 'S/p C6-7 ACDF. No urgent events overnight. Pain control ON. '
        sentences = self.rush.segToSentenceSpans(input_str)
        assert (sentences[0].begin == 0 and sentences[0].end == 14)
        assert (sentences[1].begin == 15 and sentences[1].end == 42)
        assert (sentences[2].begin == 43 and sentences[2].end == 59)

    def test3(self):
        input_str = ''' •  Coagulopathy (HCC)    



 •  Hepatic encephalopathy (HCC)    



 •  Hepatorenal syndrome (HCC)    

'''
        sentences = self.rush.segToSentenceSpans(input_str)
        assert (sentences[0].begin == 1 and sentences[0].end == 22)
        assert (sentences[1].begin == 31 and sentences[1].end == 62)
        assert (sentences[2].begin == 71 and sentences[2].end == 100)

    def test4(self):
        input_str = 'Delirium - '
        sentences = self.rush.segToSentenceSpans(input_str)
        self.printDetails(sentences, input_str)
        assert (sentences[0].begin == 0 and sentences[0].end == 8)
        pass

    def test5(self):
        input_str = "The patient complained about the TIA \n\n No memory issues. \"I \n\nOrdered the MRI scan.- "
        sentences = self.rush.segToSentenceSpans(input_str)
        self.printDetails(sentences, input_str)
        assert (sentences[0].begin == 0 and sentences[0].end == 36)
        assert (sentences[1].begin == 39 and sentences[1].end == 57)
        assert (sentences[2].begin == 58 and sentences[2].end == 84)
        pass

    def printDetails(self, sentences, input_str):
        for i in range(0, len(sentences)):
            sentence = sentences[i]
            print('assert (sentences[' + str(i) + '].begin == ' + str(sentence.begin) +
                  ' and sentences[' + str(i) + '].end == ' + str(sentence.end) + ')')
        for i in range(0, len(sentences)):
            sentence = sentences[i]
            print(input_str[sentence.begin:sentence.end])
        # self.printDetails(sentences, input_str)
        pass

    def test6(self):
        input_str = '''The Veterans Aging Cohort Study (VACS) is a large, longitudinal, observational study of a cohort of HIV infected and matched uninfected Veterans receiving care within the VA [2]. This cohort was designed to examine important health outcomes, including cardiovascular diseases like heart failure, among HIV infected and uninfected Veterans.'''
        sentences = self.rush.segToSentenceSpans(input_str)
        self.printDetails(sentences, input_str)

    def test7(self):
        input_str = '''The Veterans Aging Cohort Study (VACS) is a large, longitudinal, observational study of a cohort of HIV infected and matched uninfected Veterans receiving care within the VA [2]. This cohort was designed to examine important health outcomes, including cardiovascular diseases like heart failure, among HIV infected and uninfected Veterans.'''
        rules = []
        rules.append(r'\b(\a	0	stbegin')
        rules.append(r'\a\e	2	stend')
        rules.append(r'. +(This	0	stbegin')
        rules.append(r'](.	2	stend')
        rush = RuSH(rules, enable_logger=True)
        sentences = rush.segToSentenceSpans(input_str)
        self.printDetails(sentences, input_str)

    def test8(self):
        input_str = '''  
9.  Advair b.i.d.
10.  Xopenex q.i.d. and p.r.n.
I will see her in a month to six weeks.  She is to follow up with Dr. X before that.
'''
        self.rush = RuSH(str(os.path.join(self.pwd, 'rush_rules.tsv')), min_sent_chars=2, enable_logger=True)
        sentences = self.rush.segToSentenceSpans(input_str)
        for sent in sentences:
            print('>' + input_str[sent.begin:sent.end] + '<\n')
        assert (len(sentences) == 4)
        sent = sentences[1]
        assert (input_str[sent.begin:sent.end] == '10.  Xopenex q.i.d. and p.r.n.')

    def test9(self):
        input_str='  This is a sentence. This is another sentence.'
        self.rush = RuSH(str(os.path.join(self.pwd, 'rush_rules.tsv')), min_sent_chars=2, enable_logger=True)
        sentences = self.rush.segToSentenceSpans(input_str)
        self.printDetails(sentences, input_str)

    def test10(self):
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
        self.rush = RuSH(str(os.path.join(self.pwd, 'rush_rules.tsv')), min_sent_chars=2, enable_logger=True)
        sentences = self.rush.segToSentenceSpans(input_str)
        self.printDetails(sentences, input_str)
        assert (sentences[0].begin == 0 and sentences[0].end == 173)
        assert (sentences[1].begin == 174 and sentences[1].end == 202)
        assert (sentences[2].begin == 203 and sentences[2].end == 225)
        assert (sentences[3].begin == 226 and sentences[3].end == 258)
        assert (sentences[4].begin == 259 and sentences[4].end == 316)
        assert (sentences[5].begin == 317 and sentences[5].end == 367)
        assert (sentences[6].begin == 368 and sentences[6].end == 411)
        assert (sentences[7].begin == 412 and sentences[7].end == 447)
        assert (sentences[8].begin == 448 and sentences[8].end == 502)

    def test11(self):
        input_str = '''Patient doesn't have heart disease or high blood pressure, but their dad did have
diabetes. Pt is a 63M w/ h/o metastatic carcinoid tumor, HTN and hyperlipidemia.'''
        self.rush = RuSH(str(os.path.join(self.pwd, 'rush_rules.tsv')), min_sent_chars=2, enable_logger=True)
        sentences = self.rush.segToSentenceSpans(input_str)
        self.printDetails(sentences, input_str)
        assert (sentences[0].begin == 0 and sentences[0].end == 91)
        assert (sentences[1].begin == 92 and sentences[1].end == 162)

if __name__ == '__main__':
    unittest.main()
