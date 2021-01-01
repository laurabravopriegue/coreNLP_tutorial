import pandas as pd
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def import_and_format(dataset):
    '''

    Imports data from text file.
    Formats dataframe for getting y_pred

    :param dataset: string. either yelp or amazon
    :return: formatted dataframe
    '''

    # Import dataframe
    df = pd.read_csv('predictions/predictions_' + dataset + '.txt')

    # Sum postive and verypostitive columns to total positive score
    df['pos'] = df['pos'] + df['verypos']
    # Sum negative and verynegative columns to a total negative score
    df['neg'] = df['neg'] + df['veryneg']

    # Make sure that each review has only one sentence
    df = df[df.sent_id == 1]

    # Drop all columns except for id, total positive score, and total negative score
    df.drop(df.columns.difference(['review_id', 'neg', 'pos']), 1, inplace=True)

    # Make column review_id the index
    df = df.set_index('review_id')

    print(df.shape)

    return df

def get_y_pred(df):
    '''

    Creates the array of RNN predictions

    :param df: dataframe
    :return: array
    '''
    y_pred = []

    #If total negative score > positive score -> class 0
    #If total positive score >= negative score -> class 1
    #We dont consider neutral classes

    for index, row in df.iterrows():
        if row['neg'] > row['pos']:
            y_pred.append(0)
        else:
            y_pred.append(1)

    return y_pred

def get_y_true(dataset):
    '''

    Imports array of true values from txt file

    :param dataset: string. either yelp or amazon
    :return: array
    '''

    file= 'test_data/Y_' + dataset + '.txt'
    with open(file, 'r') as filehandle:
        y_true = json.load(filehandle)

    return y_true

def evaluation(Y_test, Y_pred, dataset):
    '''

    Evaluation of predictions against true values

    :param Y_test: array of true values
    :param Y_pred: array of preditions
    :param dataset: string. either yelp or amazon.
    '''

    #Get confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)

    #Get accuracy
    accuracy = float(cm.diagonal().sum()) / len(Y_test)
    print("Accuracy: ", accuracy)

    #Get more metrics
    more_metrics(cm)

    #Normalise and print confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    #Plot confusion matrix
    sns.heatmap(cm_normalized, annot=True, cmap='Blues')
    plt.title("Confusion Matrix " + dataset)
    plt.savefig('figs/cm_' + dataset + '.png')
    plt.show()

    return


def more_metrics(cm):
    '''

    Get precision and recall from confusion matrix values

    :param cm: nested array. confusion matrix
    '''

    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp+fp)
    recall = tp / (tp+ fn)

    print("Precision", precision)
    print("Recall", recall)

    return



datasets = ['amazon','yelp']

#Iterating through datasets
for dataset in datasets:

    print(dataset.upper())

    #Import data and format dataframe
    df = import_and_format(dataset)

    #Get array y_pred
    y_pred = get_y_pred(df)

    # Get array y_true
    y_true = get_y_true(dataset)

    #Evaluate predictions against true values
    evaluation(y_true, y_pred, dataset)












