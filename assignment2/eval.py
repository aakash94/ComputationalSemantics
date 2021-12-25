import pandas as pd
from sklearn.metrics import confusion_matrix


RES_PATH = "data/"
TSV_PATH = RES_PATH + "experiment1-dataset-color-of-concrete-objects.txt"
data_header = ["entity", "colour"]
BLACK = "black"
BLUE = "blue"
BROWN = "brown"
GREEN = "green"
GREY = "grey"
ORANGE = "orange"
PINK = "pink"
PURPLE = "purple"
RED = "red"
WHITE = "white"
YELLOW = "yellow"

COLOURS = [BLACK, BLUE, BROWN, GREEN, GREY, ORANGE, PINK, PURPLE, RED, WHITE, YELLOW]

c2i = {
    BLACK: 0,
    BLUE: 1,
    BROWN: 2,
    GREEN: 3,
    GREY: 4,
    ORANGE: 5,
    PINK: 6,
    PURPLE: 7,
    RED: 8,
    WHITE: 9,
    YELLOW: 10
}




def main():
    print("Hello World")
    df_bert = pd.read_csv("ranking_bert.csv", sep=';')
    # print(df_bert)
    df_vilbert = pd.read_csv("ranking_vilbert.csv",sep=';')
    df_bert['bert'] = df_bert['bert'].apply(lambda x: c2i[x])
    df_bert['colour'] = df_bert['colour'].apply(lambda x: c2i[x])


    true_l = df_bert['colour'].to_list()
    pred_l = df_bert['bert'].to_list()
    print(true_l)
    print(pred_l)
    matrix = confusion_matrix(true_l, pred_l)
    print(matrix)



if __name__ == '__main__':
    main()