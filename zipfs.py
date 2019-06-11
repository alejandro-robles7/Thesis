from pandas import read_csv, DataFrame
from math import pow
from collections import Counter
from json import loads
import numpy as np


def getK(q , n =3):
    return (pow(q, n) - 1) / (q - 1)


def getJ(arr, q, n=3):
    c = arr.sum()
    k = getK(q, n)
    j2 = c / k
    j1 = j2 * q
    j0 = j1 *q
    return [j0, j1, j2]

def findindex(arr, value):
    return (np.abs(arr.cumsum() - value)).argmin()

def getIndices(arr, j):
    range0 = findindex(arr, j[0]) + 1
    range1 = findindex(arr[range0 + 1:], j[1]) + 1
    return [(0, range0), (range0, range0 + range1), (range0 + range1, len(arr))]

def getSubset(arr, tup):
    return arr[tup[0]:tup[1]]

def checkCard(arr, indices):
    subs = [getSubset(arr, ind) for ind in indices]
    lens = [len(sub) for sub in subs]
    return subs, lens

def getData():
    data = []
    json_path = 'files/scrapedsites.json'
    with open(json_path) as f:
        for line in f:
            d = loads(line)
            data.append(d)

        dataFrame = DataFrame(data)
    return dataFrame

def _try(df):
    freq = df.ImpressionCount
    sites = df.site.to_numpy()

    w = freq.to_numpy()

    for q in [2, 3, 4, 5, 6]:
        j = getJ(w, q)
        s = getIndices(w, j)
        subs, lens = checkCard(w, s)
        print(q, lens)

    finaldf = df.copy()
    finaldf['SectionType'] = 0

    finaldf.loc[s[0][0]: s[0][1], 'SectionType'] = 1
    finaldf.loc[s[1][0]: s[1][1], 'SectionType'] = 2
    finaldf.loc[s[2][0]: s[2][1], 'SectionType'] = 3

    finaldf = finaldf[['site', 'TDIDCount', 'ImpressionCount', 'SectionType']]

    finaldf.to_csv('sample_output.csv', index=False)

def simple_clean(Text):
    for char in '-.,\n':
        Text = Text.replace(char, ' ')
    Text = Text.lower()
    word_list = Text.split()
    return word_list

def get_counter(words, split=False):
    if split:
        words = words.split()
    return(Counter(words).most_common())

def getJ1(w, q):
    j = getJ(w, q)
    s = getIndices(w, j)
    return getSubset(words, s[1])

# TODO Create class of this


if __name__ == '__main__':



    csv_path_cleaned = 'files/data_cleaned.txt'
    df = read_csv(csv_path_cleaned, header=None, names=['Category', 'Text'], sep=' ')
    X = df['Text']
    y = df['Category']

    df2 = getData()
    word_list = simple_clean(df2.Text[0])

    words = get_counter(word_list, split=False)
    words2 = get_counter(X[0], split=True)

    w = np.array([w[1] for w in words])

    #for q in [2, 3, 4, 5, 6]:
    for q in [2]:
        j = getJ(w, q)
        s = getIndices(w, j)
        subs, lens = checkCard(w, s)
        print(q, lens)