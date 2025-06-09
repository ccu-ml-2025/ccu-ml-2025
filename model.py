from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump #儲存模型，用以執行預測
import numpy as np
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial

### Model Evaluation Helper ###
def evaluate_binary(predicted, y_test):
    group_size = 27
    num_groups = len(predicted) // group_size
    y_pred = []
    for i in range(num_groups):
        group_prob = [p[0] for p in predicted[i*group_size:(i+1)*group_size]]
        avg_prob = np.mean(group_prob)
        y_pred.append(1 - avg_prob)

    y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
    auc = roc_auc_score(y_test_agg, y_pred, average='micro')
    return auc

def scoring_binary(estimator, X, y_true):
    y_pred = estimator.predict_proba(X)
    return evaluate_binary(y_pred, y_true)

def evaluate_multiary(predicted, y_test):
    group_size = 27
    num_groups = len(predicted) // group_size
    y_pred = []
    for i in range(num_groups):
        group_prob = predicted[i*group_size:(i+1)*group_size]
        #num_classes = len(group_prob[0])
        avg_prob = np.mean(group_prob, axis=0)
        y_pred.append(avg_prob)
        
    y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
    auc = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
    return auc

def scoring_multiary(estimator, X, y_true):
    y_pred = estimator.predict_proba(X)
    return evaluate_multiary(y_pred, y_true)

### xgboost ###
def objective_binary_xgb(params, X_train, y_train, X_test, y_test, group_size):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    if 'min_child_weight' in params:
         params['min_child_weight'] = int(params['min_child_weight'])

    clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        **params
    )

    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

    predicted_probas = clf.predict_proba(X_test)
    auc_score = evaluate_binary(predicted_probas, y_test)

    return {'loss': -auc_score, 'status': STATUS_OK, 'params': params}

def model_binary_xgb(X_train, y_train, X_test, y_test, group_size, task, max_evals = 100):
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'gamma': hp.uniform('gamma', 0, 5)
    }
    trials = Trials()
    fn_data = partial(
        objective_binary_xgb,
        X_train = X_train, y_train = y_train,
        X_test = X_test, y_test = y_test,
        group_size = group_size
    )
    best_params = fmin(
        fn = fn_data,
        space = space,
        algo = tpe.suggest,
        max_evals = max_evals,
        trials = trials,
        rstate = np.random.default_rng(42)
    )
    final_params = {
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']),
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'min_child_weight': int(best_params['min_child_weight']),
        'gamma': best_params['gamma']
    }
    clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        **final_params
    )
    clf.fit(X_train, y_train)
    predicted = clf.predict_proba(X_test)
    auc = evaluate_binary(predicted, y_test)
    print('Binary AUC (xgboost):', auc)
    dump(clf, f'./models/rf_{task}_model.joblib')

def objective_multiary_xgb(params, X_train, y_train, X_test, y_test, group_size, num_classes):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    if 'min_child_weight' in params:
        params['min_child_weight'] = int(params['min_child_weight'])


    clf = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        **params
    )
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    predicted_probas = clf.predict_proba(X_test)
    auc_score = evaluate_multiary(predicted_probas, y_test)
    return {'loss': -auc_score, 'status': STATUS_OK, 'params': params}

def model_multiary_xgb(X_train, y_train, X_test, y_test, group_size, task, max_evals = 100):
    num_classes = len(np.unique(y_train))
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'gamma': hp.uniform('gamma', 0, 5)
    }
    trials = Trials()
    fn_data = partial(
        objective_multiary_xgb,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        group_size=group_size, num_classes=num_classes
    )
    best_params = fmin(
        fn=fn_data,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )
    final_params = {
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']),
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'min_child_weight': int(best_params['min_child_weight']),
        'gamma': best_params['gamma']
    }
    clf = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        **final_params
    )
    clf.fit(X_train, y_train)
    predicted = clf.predict_proba(X_test)
    auc = evaluate_multiary(predicted, y_test)
    print('Multiary AUC(xgboost):', auc)
    dump(clf, f'./models/rf_{task}_model.joblib')

###  隨機森林 ###
def model_binary(X_train, y_train, X_test, y_test, group_size, task):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        predicted = clf.predict_proba(X_test)
        auc_score = evaluate_binary(predicted, y_test)
        print('Binary AUC (RandomForest):', auc_score)

        dump(clf, f'./models/rf_{task}_model.joblib')

# 定義多類別分類評分函數 (例如 play years、level)
def model_multiary(X_train, y_train, X_test, y_test, group_size, task):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)
        auc_score = evaluate_multiary(predicted, y_test)
        print('Multiary AUC(RandomForest):', auc_score)

        dump(clf, f'./models/rf_{task}_model.joblib')
