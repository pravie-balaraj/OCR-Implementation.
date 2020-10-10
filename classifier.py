from dataset_explore import import_data
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from feature_engineering import get_features
from sklearn.ensemble import RandomForestClassifier

# a_values = np.linspace(start=0.7, stop=0.9, num=40)
# for a in a_values:
X_train_reduced, X_test_reduced, y_train, y_test = get_features() # Values found by iterative analysis
# print(X_train_reduced.shape)



# SVM Classifier
svm = SVC(gamma=0.0051, decision_function_shape='ovr', kernel='rbf')             # has been used to enable multi class decisions
svm = svm.fit(X_train_reduced, y_train)
y_svm = svm.predict(X_test_reduced)
accuracy_svm = svm.score(X_test_reduced, y_test)
# print("PCA a =", a)
print("Accuracy for SVM Model with", accuracy_svm)
    # cm_svm = confusion_matrix(y_test, y_svm)
    # print("Confusion Matrix for SVM")
    # print(cm_svm)


# print(np.mean(X_test_reduced))

# Nearest Neighbours Classifier

# knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
# knn = knn.fit(X_train_reduced, y_train)
# y_knn = knn.predict(X_test_reduced)
# accuracy_knn = knn.score(X_test_reduced, y_test)# cm_knn = confusion_matrix(y_test, y_knn)
# print("Confusion Matrix for KNN")
# print(cm_knn)
# print("Accuracy for KNN Model")
# print(accuracy_knn)


# Logistic Regression Classifier

# clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train_reduced, y_train)
# y_clf = clf.predict(X_test_reduced)
# accuracy_clf = clf.score(X_test_reduced, y_test)
# cm_clf = confusion_matrix(y_test, y_clf)
# print("Confusion Matrix for logistic")
# print(cm_clf)
# print("Accuracy for logistic Model")
# print(accuracy_clf)
# Random Forest Classifier

# rf = RandomForestClassifier(n_estimators=200, max_depth=61)      # Accuracy depends on depth of trees too much
# rf = rf.fit(X_train_reduced, y_train)
# y_rf = rf.predict(X_test_reduced)
# accuracy_rf = rf.score(X_test_reduced, y_test)
# cm_rf = confusion_matrix(y_test, y_rf)
# print("Confusion Matrix for Random Forest")
# print(cm_rf)
# print("Accuracy for RF Model")
# print(accuracy_rf)