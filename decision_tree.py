import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from process_data import split_data, read_csv, median_cont, best_feature
import sys


def build_tree(parent, x_t, y_t, y_train):
    if len(set(y_t)) == 1:
        return y_t[0]
    elif len(y_t) == 0:
        uniques = np.unique(y_train, return_counts=True)
        return uniques[0][np.argmax(uniques[1])]
    else:
        best_column = best_feature(x_t, y_t)
        x_t_left, x_t_right, y_t_left, y_t_right = split_data(x_t, y_t, best_column)
        if len(y_t_left) == 0 or len(y_t_right) == 0:
            return parent
        uniques = np.unique(y_t, return_counts=True)
        new_parent = uniques[0][np.argmax(uniques[1])]
        left_subtree = build_tree(new_parent, x_t_left, y_t_left, y_train)
        right_subtree = build_tree(new_parent, x_t_right, y_t_right, y_train)
        decisionTree = {best_column: {}}
        decisionTree[best_column][0] = left_subtree
        decisionTree[best_column][1] = right_subtree
    return decisionTree


def predict_helper(x_t_line, decTree):
    column = list(decTree.keys())[0]
    try:
        subtree = decTree[column][x_t_line[column]]
    except:
        return 1
    if type(subtree) is dict:
        predict = predict_helper(x_t_line, subtree)
    else:
        predict = subtree
    return predict


def prediction(decTree, x_t):
    y_pred = []
    for x_t_line in x_t:
        y_pred.append(predict_helper(x_t_line, decTree))
    return y_pred


def val_matched(predict, y_vali):
    matched = 0
    for y_v in y_vali:
        if predict == y_v:
            matched += 1
    return matched


def pruning(parent, x_t, y_t, y_train, x_val, y_val):
    if len(set(y_t)) == 1:
        valid_matched = val_matched(y_t[0], y_val)
        return y_t[0], valid_matched
    elif len(y_t) == 0:
        uniques = np.unique(y_train, return_counts=True)
        pred = uniques[0][np.argmax(uniques[1])]
        valid_matched = val_matched(pred, y_val)
        return pred, valid_matched
    else:
        best_column = best_feature(x_t, y_t)
        x_t_left, x_t_right, y_t_left, y_t_right = split_data(x_t, y_t, best_column)
        x_val_left, x_val_right, y_val_left, y_val_right = split_data(x_val, y_val, best_column)
        valid_matched = val_matched(parent, y_val)
        if len(y_t_left) == 0 or len(y_t_right) == 0:
            return parent, valid_matched
        uniques = np.unique(y_t, return_counts=True)
        new_parent = uniques[0][np.argmax(uniques[1])]
        left_subtree, left_matched = pruning(new_parent, x_t_left, y_t_left, y_train, x_val_left, y_val_left)
        right_subtree, right_matched = pruning(new_parent, x_t_right, y_t_right, y_train, x_val_right, y_val_right)
        if valid_matched >= left_matched + right_matched:
            return parent, valid_matched
        decisionTree = {best_column: {}}
        decisionTree[best_column][0] = left_subtree
        decisionTree[best_column][1] = right_subtree
    return decisionTree, left_matched + right_matched


def part_c(x_train, y_train, x_test, y_test, x_val, y_val):
    n_estimators1 = [50 + 100 * i for i in range(0, 5)]
    max_feature1 = [0.1 + 0.2 * i for i in range(0, 5)]
    min_samples_split1 = [2 + 2 * i for i in range(0, 5)]
    clfcv = RandomForestClassifier(oob_score=True)
    param_grid = {'n_estimators': n_estimators1, 'max_features': max_feature1, 'min_samples_split': min_samples_split1}
    search = GridSearchCV(estimator=clfcv, param_grid=param_grid, cv=5)
    search.fit(x_train, y_train)
    search.predict(x_test)
    search.predict(x_train)
    search.predict(x_val)
    print(search.best_params_)

def part_d(x_train, y_train, x_test, y_test, x_val, y_val):
    clfcv = RandomForestClassifier(n_estimators=100)
    clfcv.fit(x_train, y_train)
    accuracy_test = metrics.accuracy_score(clfcv.predict(x_test), y_test)
    print(accuracy_test)

part = sys.argv[1]
train_file = sys.argv[2]
val_file = sys.argv[3]
test_file = sys.argv[4]
output_file = sys.argv[5]
x_train, y_train = read_csv(train_file)
x_test, y_test = read_csv(test_file)
x_val, y_val = read_csv(val_file)
x_train = median_cont(x_train)
x_test = median_cont(x_test)
x_val = median_cont(x_val)

if part == '1':
    finalTree = build_tree(None, x_train, y_train, y_train)
    y_pred = prediction(finalTree, x_test)

elif part == '2':
    # finalTree, temp_count = pruning(None, x_train, y_train, y_train, x_val, y_val)
    finalTree = build_tree(None, x_train, y_train, y_train)
    y_pred = prediction(finalTree, x_test)

f = open(output_file, 'w')
for i in y_pred:
    f.write(str(i))
    f.write('\n')
f.close()