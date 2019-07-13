#Training an Logistic Regression Model on TF-IDF vectors

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

csv_path_cleaned = 'files/data_cleaned.txt'

df = pd.read_csv(csv_path_cleaned, header = None, names = ['Category', 'Text'], sep =' ')
X = df['Text']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

logreg_model = Pipeline((('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf',
                        LogisticRegression(multi_class='multinomial', solver='lbfgs'
                        ))))

#train the model
logreg_model.fit(X_train, y_train)

y_pred_linear = logreg_model.predict(X_test)

print('accuracy linear model %s' % accuracy_score(y_pred_linear, y_test))

print("report linear model", classification_report(y_test, y_pred_linear,target_names=sgd_linear.classes_))
