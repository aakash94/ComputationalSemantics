import pandas as pd
from io import open
from conllu import parse_incr


class TreeParse:
    '''
    Install CoNLL-U Parser to use this.
    To install you may use 'pip install conllu'
    Feel free to edit and make changes as you see fit.
    You can look up more documentation in the following places
        https://pypi.org/project/conllu/
        https://github.com/EmilStenstrom/conllu/blob/master/README.md

    '''

    def __init__(self, conllu_path):
        '''
        :param conllu_path: Path to the Conllu File that you want to process.
        :type conllu_path: str
        '''
        self.conllu_data = open(conllu_path, "r", encoding="utf-8")
        self.header = ['context_before', 'target_sentence', 'context_after']
        self.df = pd.DataFrame(columns=self.header)

    def print_all_lines(self):
        '''
        You may use this to explore the conllu file.
        :return: Nothing
        :rtype: None
        '''
        for sentence in parse_incr(self.conllu_data):
            print(sentence.metadata['text'])

    def remove_tabs(self, strng):
        '''

        :param strng: the input string from which all tab characters must be removed.
        :type strng:str
        :return: returns the same string with all the tabs replaced to space
        :rtype: str
        '''
        return strng.replace("\t", " ")

    def get_context_lines(self, list7):
        '''
        :param list7:
        :type list7:list
        :return: returns 1st 3 and last 3 sentences from the token list without any tabs
        :rtype: tuple
        '''
        lines_before = list7[0].metadata['text'] + " " + list7[1].metadata['text'] + " " + list7[2].metadata['text']
        lines_after = list7[4].metadata['text'] + " " + list7[5].metadata['text'] + " " + list7[6].metadata['text']
        return self.remove_tabs(lines_before), self.remove_tabs(lines_after)

    def get_lines(self,
                  word="also",
                  upos_req="verb",
                  max_count=1024,
                  compare_lowercase=True,
                  compare_title=True,
                  compare_uppercase=True):
        '''
        :param word: The word to be searched in the sentence
        :type word: str
        :param max_count: Maximum number of lines that should be returned
        :type max_count: int
        :param compare_lowercase: True if the lowercase version of the original word should be searched
        :type compare_lowercase: bool
        :param compare_title: True if the title version (only first letter capital) of the original word should be searched
        :type compare_title: bool
        :param compare_uppercase: True if the uppercase version of the original word should be searched
        :type compare_uppercase: bool
        :return: returns a dataframe which has target sentence and and surrounding context
        :rtype: dataframe
        '''
        context_before_list = []
        sentence_list = []
        context_after_list = []

        last7 = []

        for sentence in parse_incr(self.conllu_data):

            last7.append(sentence)

            if len(last7) < 7:
                continue

            elif len(last7) > 7:
                last7.pop(0)

            if len(sentence_list) == max_count:
                break

            target_sentence = last7[3]

            if target_sentence.filter(lemma=word,upos=upos_req.upper()).__len__() > 0:
                l_before, l_after = self.get_context_lines(last7)
                sentence_string = self.remove_tabs(target_sentence.metadata['text'])
                sentence_list.append(sentence_string)
                context_before_list.append(l_before)
                context_after_list.append(l_after)
            

        self.df['context_before'] = context_before_list
        self.df['target_sentence'] = sentence_list
        self.df['context_after'] = context_after_list

        return self.df

    def dump_tsv(self, dump_path, max_count=128):
        '''

        :param dump_path: the path where the file will be dumped
        :type dump_path: str
        :param max_count: how many random samples to dump
        :type max_count: int
        :return: none
        :rtype: none
        '''
        df_count = len(self.df.index)
        sample_count = min(max_count, df_count)
        sampled_lines = self.df.sample(sample_count)
        sampled_lines.to_csv(dump_path, sep="\t", index=False)
        print("Dumped ", sample_count, " lines to ",dump_path)


def demo():
    print("Hello World, this function demonstrates how to use TreeParse")
    print("install conllu to use this script. use 'pip install conllu'")
    print("The output csv is TAB separated and NOT comma separated")
    print("All the best!\n\n\n")

    CONLLU_PATH = "data/Universal Dependencies 2.8.1/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-test.conllu"
    WORD = "result"
    upos_req="verb"
    OUTPUT_PATH = "data/output.csv"

    # To use, create an instance of the class, and pass the path to the CONLLU file
    tp = TreeParse(conllu_path=CONLLU_PATH)
    tp.get_lines(word=WORD,upos_req=upos_req)
    tp.dump_tsv(dump_path=OUTPUT_PATH)


if __name__ == "__main__":
    demo()
