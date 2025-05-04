from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump #儲存模型，用以執行預測
import numpy as np

### KNN ###
def model_binary_knn(X_train, y_train, X_test, y_test, group_size, task):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    predicted = clf.predict_proba(X_test)
    predicted = [p[0] for p in predicted]

    num_groups = len(predicted) // group_size
    if sum(predicted[:group_size]) / group_size > 0.5:
        y_pred = [max(predicted[i*group_size:(i+1)*group_size]) for i in range(num_groups)]
    else:
        y_pred = [min(predicted[i*group_size:(i+1)*group_size]) for i in range(num_groups)]
    y_pred = [1 - x for x in y_pred]
    y_test_agg = [y_test[i*group_size] for i in range(num_groups)]

    auc = roc_auc_score(y_test_agg, y_pred, average='micro')
    print('Binary AUC (KNN):', auc)

    dump(clf, f'./models/rf_{task}_model.joblib')

def model_multiary_knn(X_train, y_train, X_test, y_test, group_size, task):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    predicted = clf.predict_proba(X_test)

    num_groups = len(predicted) // group_size
    y_pred = []
    for i in range(num_groups):
        group_pred = predicted[i*group_size:(i+1)*group_size]
        num_classes = len(group_pred[0])
        class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
        chosen_class = np.argmax(class_sums)
        candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
        best_instance = np.argmax(candidate_probs)
        y_pred.append(group_pred[best_instance])

    y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
    auc = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
    print('Multiary AUC (KNN):', auc)

    dump(clf, f'./models/rf_{task}_model.joblib')

###  隨機森林 ###
def model_binary(X_train, y_train, X_test, y_test, group_size, task):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        predicted = clf.predict_proba(X_test)
        # 取出正類（index 0）的概率
        predicted = [predicted[i][0] for i in range(len(predicted))]
        
        
        num_groups = len(predicted) // group_size 
        if sum(predicted[:group_size]) / group_size > 0.5:
            y_pred = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        else:
            y_pred = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        
        y_pred  = [1 - x for x in y_pred]
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print('Binary AUC (RandomForest):', auc_score)

        dump(clf, f'./models/rf_{task}_model.joblib')

# 定義多類別分類評分函數 (例如 play years、level)
def model_multiary(X_train, y_train, X_test, y_test, group_size, task):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = []

        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            num_classes = len(np.unique(y_train))
            # 對每個類別計算該組內的總機率
            class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
            best_instance = np.argmax(candidate_probs)
            y_pred.append(group_pred[best_instance])
        
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        print('Multiary AUC(RandomForest):', auc_score)

        dump(clf, f'./models/rf_{task}_model.joblib')