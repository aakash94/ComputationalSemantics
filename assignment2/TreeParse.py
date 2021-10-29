from io import open
from conllu import parse_incr
import pandas as pd


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
        self.conllu_data2 = open(conllu_path, "r", encoding="utf-8")
        self.before_context=[]
        self.after_context=[]
        self.sentence_list = []
        self.parse_incr_val=parse_incr(self.conllu_data2)
        self.all_content=[s.metadata['text'] for s in self.parse_incr_val]

    def print_all_lines(self):
        '''
        You may use this to explore the conllu file.
        :return: Nothing
        :rtype: None
        '''
        for sentence in parse_incr(self.conllu_data):
            print(sentence.metadata['text'])

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
        
        
        
        
        def save_data(self,sentence,counterPlus):
            print(counterPlus)
            self.sentence_list.append(sentence.metadata['text'])
            if counterPlus>=3:
                self.before_context.append(self.all_content[counterPlus-3]+self.all_content[counterPlus-2]+self.all_content[counterPlus-1])
            
            elif counterPlus==2:
                self.before_context.append(self.all_content[counterPlus-2]+self.all_content[counterPlus-1])
            
            elif counterPlus==1:
                self.before_context.append(self.all_content[counterPlus-1])
            
            elif counterPlus==0:
                self.before_context.append(" ")
                
            if counterPlus<=len(self.all_content)-3:
                self.after_context.append(self.all_content[counterPlus+1]+self.all_content[counterPlus+2]+self.all_content[counterPlus+3])
            
            elif counterPlus==len(self.all_content):
                self.after_context.append(" ")
            
            elif counterPlus==len(self.all_content)-1:
                self.after_context.append(self.all_content[counterPlus+1])
            
            elif counterPlus==len(self.all_content)-2:
                self.after_context.append(self.all_content[counterPlus+1]+self.all_content[counterPlus+2])
                
                
        final_context={}
        counter=0
        for sentence in parse_incr(self.conllu_data):
            
            
            if len(self.sentence_list) == max_count:
                break

            if sentence.filter(form=word).__len__() > 0:
                save_data(self,sentence,counter)
                
            elif compare_lowercase and sentence.filter(form=word.lower()).__len__() > 0:
                save_data(self,sentence,counter)
                
            elif compare_title and sentence.filter(form=word.title()).__len__() > 0:
                save_data(self,sentence,counter)
                
            elif compare_uppercase and sentence.filter(form=word.upper()).__len__() > 0:
                save_data(self,sentence,counter)
                
            counter+=1
            
            
           
        final_context["context_before"]=self.before_context
        final_context["target_sentence"]=self.sentence_list
        final_context["context_after"]=self.after_context
        return final_context


def demo():
    print("Hello World, this function demonstrates how to use TreeParse")
    print("install conllu to use this script. use 'pip install conllu'")
    print("All the best!\n\n\n")

    CONLLU_PATH = "data/Universal Dependencies 2.8.1/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-test.conllu"
    WORD = "peppermint"

    # To use, create an instance of the class, and pass the path to the CONLLU file
    tp = TreeParse(conllu_path=CONLLU_PATH)

    # To view all the lines in the file
    # tp.print_all_lines()

    # To get lines containing the word, simply pass the word to the get_lines word.
    sentence_list = tp.get_lines(word=WORD)

    # You can do whatever required with the lines here.
    print(len(sentence_list["target_sentence"]), " Sentences found for the word ", WORD)
    data_frame=pd.DataFrame(sentence_list)
    data_frame.to_csv("sentences.csv",index=0)
    for s in sentence_list["target_sentence"]:
        print(s)


if __name__ == "__main__":
    demo()
