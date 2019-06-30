from pandas import DataFrame
from math import pow
from collections import Counter
from json import loads
from nltk.corpus import stopwords
from langdetect import detect
import numpy as np

def getData(path='scrapedsites.json'):
    data = []
    with open(path) as f:
        for line in f:
            d = loads(line)
            data.append(d)

        dataFrame = DataFrame(data)
    return dataFrame

class Zipf:

    target_column = ''
    vector_column = 'Counter_Text'
    output_column = 'Clean_Text'


    def __init__(self, q=2, lower=True):
        self.q = q
        self.lower = lower

    def fit(self, dataframe, target_column='Text'):
        self.target_column = target_column
        self.dataframe = dataframe

    def transform(self, output_column='Clean_Text'):
        self.output_column = output_column
        self._vectorize()
        self._cleanText()

    def filter_by_language(self, lang='en', col_name=None):
        if col_name:
            coltarget=col_name
        else:
            coltarget=self.target_column

        bool_index = [False] * self.dataframe.shape[0]
        for i, text in enumerate(self.dataframe[coltarget]):
            try:
                bool_index[i] = detect(text) == lang
            except:
                pass
        self.dataframe = self.dataframe.loc[bool_index]

    def _vectorize(self):
        clean_text = []
        if self.lower:
            self.dataframe[self.target_column] = self.dataframe[self.target_column].str.lower()
        for text in self.dataframe[self.target_column]:
            try:
                temp = Zipf.getJ1(text, self.q, True)
            except:
                temp = []
            clean_text.append(temp)
        self.dataframe[self.vector_column] = clean_text


    def _cleanText(self):
        clean_text = []
        for text_vector in self.dataframe[self.vector_column]:
            try:
                words = self.get_strings(text_vector)
                no_stops = self.remove_stopwords(words)
                clean_row = ' '.join(no_stops)
            except:
                clean_row = np.nan
            clean_text.append(clean_row)
        self.dataframe[self.output_column] = clean_text



    @staticmethod
    def getK(q, n=3):
        return (pow(q, n) - 1) / (q - 1)

    @staticmethod
    def getJ(arr, q, n=3):
        c = arr.sum()
        k = Zipf.getK(q, n)
        j2 = c / k
        j1 = j2 * q
        j0 = j1 * q
        return [j0, j1, j2]

    @staticmethod
    def findindex(arr, value):
        return (np.abs(arr.cumsum() - value)).argmin()

    @staticmethod
    def getIndices(arr, j):
        range0 = Zipf.findindex(arr, j[0]) + 1
        range1 = Zipf.findindex(arr[range0 + 1:], j[1]) + 1
        return [(0, range0), (range0, range0 + range1), (range0 + range1, len(arr))]

    @staticmethod
    def getSubset(arr, tup):
        return arr[tup[0]:tup[1]]

    @staticmethod
    def checkCard(arr, indices):
        subs = [Zipf.getSubset(arr, ind) for ind in indices]
        lens = [len(sub) for sub in subs]
        return subs, lens

    @staticmethod
    def getJ1(text, q, split=False):
        words_dict = Zipf.get_counter(text, split)
        word_counts = np.array([word[1] for word in words_dict])
        j = Zipf.getJ(word_counts, q)
        s = Zipf.getIndices(word_counts, j)
        return Zipf.getSubset(words_dict, s[1])

    @staticmethod
    def get_counter(words, split=False):
        if split:
            words = words.split()
        return Counter(words).most_common()

    @staticmethod
    def get_strings(arr, index=0):
        return [w[index] for w in arr]

    @staticmethod
    def remove_stopwords(words):
        return [word for word in words if word not in stopwords.words('english')]

if __name__ == '__main__':

    path = 'files/scrapedsites.json'
    df = getData(path)
    z = Zipf(q=2)
    z.fit(df)
    z.transform()



