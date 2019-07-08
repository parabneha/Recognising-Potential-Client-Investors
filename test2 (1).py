from sklearn.feature_extraction.text import *
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics import classification_report 
from sklearn import svm 
from sklearn.model_selection import train_test_split
import csv
casefile = csv.reader(open('bank2.csv','r'))


data = []
labels = []
for row in casefile:
    data.append(row[0:15])
    labels.append(row[16])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3,random_state=2)

print(X_test)

#print(X_train)
#print(y_train)
# Create feature vectors 
#vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
# Train the feature vectors
#train_vectors = vectorizer.fit_transform(X_train)
#print(train_vectors)
# Apply model on test data 
#test_vectors = vectorizer.transform(X_test)
#print(test_vectors)
#perform=vectorizer.transform(ip)


# Perform classification with SVM, kernel=linear 
model = svm.SVC(kernel='linear') 
model.fit(X_train, y_train) 
prediction = model.predict(X_test)
print(prediction)


#print(model.predict(ip))
print (classification_report(y_test, prediction))

