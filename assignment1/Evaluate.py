import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, \
    ConfusionMatrixDisplay

REPORT_PATH = "data/result.csv"


def sentiment_to_score(senti):
    if senti == 'neg':
        return 0
    else:
        return 1

def get_report(file_name):
    df_report = pd.read_csv(file_name)
    print("Report has", len(df_report), " lines")
    df_report = df_report.iloc[:, 1:]
    df_report['gold_values'] = df_report.gold_values.apply(sentiment_to_score)
    df_report['predictions'] = df_report.predictions.apply(sentiment_to_score)
    return df_report

def get_metrics(y_true, y_pred):
    a_score = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    ps = precision_score(y_true, y_pred)
    rs = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("accuracy score = ", a_score)
    print("Precision score = ", ps)
    print("Recall score = ", rs)
    print("F1 score = ", f1)
    print("Confusion Matrix = \n", cm, "\n" )
    cm = confusion_matrix(y_true, y_pred,)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()



def main():
    print("Hello World")
    df = get_report(REPORT_PATH)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df[:3])

    # gv = df.gold_values.to_numpy()
    # p = df.predictions.to_numpy()
    #
    # get_metrics(y_true=gv, y_pred=p)

    # print(gv)
    # print(type(p))

    only_wrongs = df.loc[df['gold_values'] != df['predictions']]
    only_wrongs.to_csv(r'C:\Users\Aakash\Workspace\Term1\ComputationalSemantics\assignment1\data\only_wrongs.csv',index=False)



if __name__ == "__main__":
    main()
