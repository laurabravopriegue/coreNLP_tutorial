import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string


def importCSV(dataset):
    '''
    imports csv from .txt file as a dataframe

    :param dataset: string. either "yelp" or "amazon"
    :return: dataframe
    '''

    path = 'raw_data/raw_' + dataset + '.txt'

    dataframe = pd.read_csv(path, sep='\t', names = ['review', 'target'])

    print("Dataframe imported ", dataset)

    print("Size " + dataset, dataframe.shape)
    print(dataframe.head())

    print("How many instances of each target in the whole dataset: ")
    print(dataframe['target'].value_counts())

    return dataframe

def remove_punctuations(text):
    '''
    removes punctuation from text

    :param text: string
    :return: string
    '''
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def formatInput(df, dataset):
    '''

    - formatting of reviews for RNN processing -> outputs a .txt file
    - formatting of targets for evaluation -> outputs a .txt file

    :param df: dataframe
    :param dataset: string
    '''

    df["review"] =df['review'].apply(remove_punctuations)

    X = df.iloc[:, :-1].values
    Y = df['target'].to_list()

    flatten_x = [val for sublist in X for val in sublist]

    joined_string = "\n".join(flatten_x)

    path = "test_data/X_" + dataset +".txt"
    print("Writes reviews to file ", path)
    text_file = open(path, "w")
    text_file.write(joined_string)
    text_file.close()

    path = "test_data/Y_" + dataset + ".txt"
    print("Writes targets to .txt file ", path)
    print(Y, file=open(path, 'w'))

    return



def explore_length(df1, df2, dataset1, dataset2):
    '''

    Stats on the length of reviews for both datasets
    Calls function for printing stats
    Calls function for plotting histogram

    :param df1: dataframe 1
    :param df2: dataframe 2
    :param dataset1: string
    :param dataset2: string
    '''
    reviews1 = df1['review'].to_list()
    reviews2 = df2['review'].to_list()

    lens1 = [len(x.split()) for x in reviews1]
    lens2 = [len(x.split()) for x in reviews2]

    len_stats(lens1, dataset1)
    len_stats(lens2, dataset2)

    histo(lens1, lens2)

    return

def len_stats(len_list, dataset):
    '''

    Prints stats of lens of reviews

    :param len_list: list of lens of reviews
    :param dataset: string - name of dataset
    '''
    print("Len exploration ", dataset)

    print("Max length ", max(len_list))
    print("Min length ", min(len_list))
    print("Std Dev ", np.std(len_list))
    print("Avg length ", sum(len_list) / len(len_list))
    return

def histo(data1, data2):
    '''
    Plots histogram of lens of reviews

    :param data1: array. data from dataset 1
    :param data2: array. data from dataset 2
    '''
    bins = 19

    plt.style.use('bmh')

    plt.hist(data1, bins, alpha=0.5, label='yelp', density=True, range=[0, 32], edgecolor='grey')
    plt.hist(data2, bins, alpha=0.5, label='amazon', density=True, range=[0, 32], edgecolor='grey')
    plt.legend(loc='upper right')
    plt.xlabel('Number of sentences per review')
    plt.ylabel('Density')
    plt.savefig('figs/histo.png')
    plt.show()

    return

#The two possible datasets
raw_datasets = ["yelp", "amazon"]

#Creating the dataframes
df_yelp = importCSV(raw_datasets[0])
df_amazon = importCSV(raw_datasets[1])

#Formatting input for modelling and evaluation
formatInput(df_yelp, 'yelp')
formatInput(df_amazon, 'amazon')

#Exploring the distribution of review length
explore_length(df_yelp, df_amazon, raw_datasets[0], raw_datasets[1])