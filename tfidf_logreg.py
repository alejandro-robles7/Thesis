#Training an Logistic Regression Model on TF-IDF vectors

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from glove_vectors_logreg import *

def model_pipe(binary=True):
    if binary:
        model = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', LogisticRegression()
                           )])
    else:
        model = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs')
                           )])
    return model

def train_pipeline(x, y, model):
    X_train, X_test, y_train, y_test = split_dataset(x, y)
    model = model.fit(X_train, y_train)
    return calc_accuracy(model, X_test, y_test), model


if __name__ == '__main__':

    csv_path_cleaned = 'files/data_cleaned.txt'
    binary_flag = True

    X, y = load_data(csv_path_cleaned)

    if binary_flag:
        sports_ind, other_ind = binarize_ind(y, one_vs_all=True)
        y = mask_label(y, other_ind)

    logreg_model = model_pipe(binary_flag)
    accuracy, model = train_pipeline(X, y, logreg_model)
