from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle as pk

def SVM(X_train_pca, Y_train, name):
    print("Running SVM")
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],  
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 
                'kernel': ['rbf']} 
    svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                    param_grid, cv=5, refit = True)
    svm = svm.fit(X_train_pca, Y_train)
    file = 'PJ1\\model\\' + name
    pk.dump(svm, open(file, 'wb'))
    print("Best estimator found by grid search:")
    print(svm.best_estimator_)
    print("Done SVM !!")
    print('___________________________________________')