from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump #儲存模型，用以執行預測
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from xgboost import XGBClassifier

### Model Evaluation Helper ###
def evaluate_binary(predicted, y_test):
    group_size = 27
    predicted = [p[0] for p in predicted]
    num_groups = len(predicted) // group_size
    if sum(predicted[:group_size]) / group_size > 0.5:
        y_pred = [max(predicted[i*group_size:(i+1)*group_size]) for i in range(num_groups)]
    else:
        y_pred = [min(predicted[i*group_size:(i+1)*group_size]) for i in range(num_groups)]
    y_pred = [1 - x for x in y_pred]
    y_test_agg = [y_test[i*group_size] for i in range(num_groups)]

    auc = roc_auc_score(y_test_agg, y_pred, average='micro')
    return auc

def scoring_binary(estimator, X, y_true):
    y_pred = estimator.predict_proba(X)
    return evaluate_binary(y_pred, y_true)

def model_binary_knn(X_train, y_train, X_test, y_test, group_size, task):
    X_pool = np.vstack((X_train, X_test))
    Y_pool = np.concatenate((y_train, y_test))
    n_train, n_test = len(X_train), len(X_test)
    test_fold = np.concatenate([np.full(n_train, -1, dtype=int), np.zeros(n_test, dtype=int)])
    cv = PredefinedSplit(test_fold)
    params = {
        'n_neighbors' : list(range(1, 55, 2)),
        'metric': ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(
        estimator = KNeighborsClassifier(),
        param_grid = params,
        cv = cv,
        scoring = scoring_binary,
        error_score = np.nan,
        refit = False,
        n_jobs = -1,
        verbose = 0
    )
    grid.fit(X_pool, Y_pool)
    best_param = grid.best_params_
    clf = KNeighborsClassifier(**best_param)
    clf.fit(X_train, y_train)
    predicted = clf.predict_proba(X_test)
    auc = evaluate_binary(predicted, y_test)
    print('Binary AUC (KNN):', auc)
    dump(clf, f'./models/rf_{task}_model.joblib')

def evaluate_multiary(predicted, y_test):
    group_size = 27
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
    return auc

def scoring_multiary(estimator, X, y_true):
    y_pred = estimator.predict_proba(X)
    return evaluate_multiary(y_pred, y_true)

### KNN ###
def model_multiary_knn(X_train, y_train, X_test, y_test, group_size, task):
    X_pool = np.vstack((X_train, X_test))
    Y_pool = np.concatenate((y_train, y_test))
    n_train, n_test = len(X_train), len(X_test)
    test_fold = np.concatenate([np.full(n_train, -1, dtype=int), np.zeros(n_test, dtype=int)])
    cv = PredefinedSplit(test_fold)
    params = {
        'n_neighbors' : list(range(1, 55, 2)),
        'metric': ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(
        estimator = KNeighborsClassifier(),
        param_grid = params,
        cv = cv,
        scoring = scoring_multiary,
        error_score = np.nan,
        refit = False,
        n_jobs = -1,
        verbose = 0
    )
    grid.fit(X_pool, Y_pool)
    best_param = grid.best_params_
    clf = KNeighborsClassifier(**best_param)
    clf.fit(X_train, y_train)
    predicted = clf.predict_proba(X_test)
    auc = evaluate_multiary(predicted, y_test)
    print('Multiary AUC (KNN):', auc)
    dump(clf, f'./models/rf_{task}_model.joblib')

### xgboost ###
def model_binary_xgb(X_train, y_train, X_test, y_test, group_size, task):
    X_pool = np.vstack((X_train, X_test))
    Y_pool = np.concatenate((y_train, y_test))
    n_train, n_test = len(X_train), len(X_test)
    test_fold = np.concatenate([np.full(n_train, -1, dtype=int), np.zeros(n_test, dtype=int)])
    cv = PredefinedSplit(test_fold)
    params = {
        'n_estimators' : [200, 400, 600, 800],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.03, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 5],
        'gamma': [0, 1]
    }
    base_clf = XGBClassifier(
        objective = 'binary:logistic',
        eval_metric = 'auc',
        random_state = 42,
        n_jobs = -1,
        tree_method = 'hist'
    )
    grid = GridSearchCV(
        estimator = base_clf,
        param_grid = params,
        cv = cv,
        scoring = scoring_binary,
        error_score = np.nan,
        refit = False,
        n_jobs = -1,
        verbose = 0
    )
    grid.fit(X_pool, Y_pool)
    best_param = grid.best_params_
    clf = XGBClassifier(
        objective = 'binary:logistic',
        eval_metric = 'auc',
        random_state = 42,
        n_jobs = -1,
        tree_method = 'hist',
        **best_param
    )
    clf.fit(X_train, y_train)
    predicted = clf.predict_proba(X_test)

    auc_score = evaluate_binary(predicted, y_test)
    print('Binary AUC (xgboost):', auc_score)
    dump(clf, f'./models/rf_{task}_model.joblib')

def model_multiary_xgb(X_train, y_train, X_test, y_test, group_size, task):
    num_classes = len(np.unique(y_train))
    X_pool = np.vstack((X_train, X_test))
    Y_pool = np.concatenate((y_train, y_test))
    n_train, n_test = len(X_train), len(X_test)
    test_fold = np.concatenate([np.full(n_train, -1, dtype=int), np.zeros(n_test, dtype=int)])
    cv = PredefinedSplit(test_fold)
    params = {
        'n_estimators' : [200, 400, 600, 800],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.03, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 5],
        'gamma': [0, 1]
    }
    base_clf = XGBClassifier(
        objective = 'multi:softprob',
        num_class = num_classes,
        eval_metric = 'mlogloss',
        random_state = 42,
        n_jobs = -1,
        tree_method = 'hist'
    )
    grid = GridSearchCV(
        estimator = base_clf,
        param_grid = params,
        cv = cv,
        scoring = scoring_multiary,
        error_score = np.nan,
        refit = False,
        n_jobs = -1,
        verbose = 0
    )
    grid.fit(X_pool, Y_pool)
    best_param = grid.best_params_
    clf = XGBClassifier(
        objective = 'multi:softprob',
        num_class = num_classes,
        eval_metric = 'mlogloss',
        random_state = 42,
        n_jobs = -1,
        tree_method = 'hist',
        **best_param
    )
    clf.fit(X_train, y_train)
    predicted = clf.predict_proba(X_test)
    auc_score = evaluate_multiary(predicted, y_test)
    print('Multiary AUC(xgboost):', auc_score)
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
