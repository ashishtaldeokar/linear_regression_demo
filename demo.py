import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#read data
dataframe = pd.read_fwf('brain_body.txt')
challenge_dataset = pd.read_csv('challenge_dataset.txt')

x_challege = challenge_dataset[['Brain']]
y_challege = challenge_dataset[['Body']]

X_train, X_test, y_train, y_test = train_test_split(x_challege, y_challege, test_size=0.4, random_state=0)

#train model on data
cross_clf = linear_model.LinearRegression()
cross_clf.fit(X_train, y_train)
clf = linear_model.LinearRegression()
clf.fit(x_challege, y_challege)
print "--------CLASSIFIER ERROR--------"
print "error (%): ",(1 - clf.score(X_train,y_train))*100
print "cross validation error (%): ",(1 - cross_clf.score(X_test,y_test))*100
print "--------MEAN SQUARED ERROR--------"
print "rms error: ",mean_squared_error(y_challege,clf.predict(x_challege))
print "cross validation rms error: ",mean_squared_error(y_test,cross_clf.predict(X_test))


#visualize results
plt.scatter(X_test, y_test, c = 'red', label="testing data")
plt.scatter(X_train, y_train, c='blue', label="training data")

plt.plot(X_test, cross_clf.predict(X_test), label="cross validation")
plt.plot(x_challege, clf.predict(x_challege), label="non cross validation")
plt.legend()

plt.show()
