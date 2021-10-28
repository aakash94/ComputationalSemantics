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

    def get_lines(self, word="also", max_count=128, compare_lowercase=True, compare_title=True, compare_uppercase=True):
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
        :return: List of strings, where each string is a line in the conllu file that contains the word
        :rtype: list
        '''
        sentence_list = []

        for sentence in parse_incr(self.conllu_data):
            if len(sentence_list) == max_count:
                break

            if sentence.filter(form=word).__len__() > 0:
                sentence_list.append(sentence.metadata['text'])

            elif compare_lowercase and sentence.filter(form=word.lower()).__len__() > 0:
                sentence_list.append(sentence.metadata['text'])

            elif compare_title and sentence.filter(form=word.title()).__len__() > 0:
                sentence_list.append(sentence.metadata['text'])

            elif compare_uppercase and sentence.filter(form=word.upper()).__len__() > 0:
                sentence_list.append(sentence.metadata['text'])

        return sentence_list


def demo():
    print("Hello World, this function demonstrates how to use TreeParse")
    print("install conllu to use this script. use 'pip install conllu'")
    print("All the best!\n\n\n")

    CONLLU_PATH = "data/Universal Dependencies 2.8.1/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-test.conllu"
    WORD = "peppermint"

    # To use, create an instance of the class, and pass the path to the CONLLU file
    tp = TreeParse(conllu_path=CONLLU_PATH)

    # To get lines containing the word, simply pass the word to the get_lines word.
    sentence_list = tp.get_lines(word=WORD)

    # You can do whatever required with the lines here.
    print(len(sentence_list), " Sentences found for the word ", WORD)
    for s in sentence_list:
        print(s)


if __name__ == "__main__":
    demo()
