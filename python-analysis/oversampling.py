import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, \
    precision_score, recall_score
import joblib
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

ITERATIONS = 20

def save_timeouts(dict_timeouts, file_name):
    directory = 'TIMEOUTS/'
    file_name = directory + file_name + ".csv"
    aux_df = pd.DataFrame([dict_timeouts])
    aux_df.to_csv(file_name, index=False)


def get_classification_report(y_test, predictions):
    print(classification_report(y_test, predictions))
    print('Confusion Matrix')
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    # ax = sns.heatmap(cm, linewidth=0.5, annot=True)
    # sns.color_palette("Paired")


def get_metrics(y_test, predictions):
    # ACCURACY => ACC = (TP + TN) / (TP + TN + FP + FN)
    # SENSITIVITY => TPR = TP / (TP + FN)
    # SPECIFICITY => TNR = TN / (TN + FP)
    # tp, fp, fn, tn = confusion_matrix(y_test, predictions).ravel()
    cm1 = confusion_matrix(y_test, predictions)
    total1 = sum(sum(cm1))

    accuracy = ((cm1[0, 0] + cm1[1, 1]) / total1) * 100
    sensitivity = (cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])) * 100
    specificity = (cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])) * 100
    balanced_accuracy = (sensitivity + specificity) / 2

    print('Accuracy: %.2f' % accuracy + '%')
    print("Balanced Accuracy: %.2f" % balanced_accuracy + '%')
    print("Sensitivity: %.2f" % sensitivity + '%')
    print("Specifity: %.2f" % specificity + '%')
    return [accuracy, sensitivity, specificity, balanced_accuracy]

def get_results_for_algorithm(algorithm_name, x_train, y_train, x_test, y_test):
    import time
    if algorithm_name == 'LR':
        start = time.time()
        clf = LogisticRegression(solver='lbfgs', max_iter=10000)  # class_weight={0:1, 1:500})
        clf.fit(x_train, y_train)
        end = time.time()
    elif algorithm_name == 'DT':
        start = time.time()
        clf = DecisionTreeClassifier()  # class_weight={0:1, 1:500})
        clf.fit(x_train, y_train)
        end = time.time()
    elif algorithm_name == 'RF':
        start = time.time()
        clf = RandomForestClassifier(n_estimators=300)  # class_weight={0:1, 1:500})
        clf.fit(x_train, y_train)
        end = time.time()
    elif algorithm_name == 'SVM':
        start = time.time()
        clf = svm.SVC(kernel='linear', probability=True)  # class_weight={0:1, 1:500})
        clf.fit(x_train, y_train)
        end = time.time()
    elif algorithm_name == 'GNB':
        start = time.time()
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        end = time.time()
    elif algorithm_name == 'BNB':
        start = time.time()
        clf = BernoulliNB(binarize=0.0)
        clf.fit(x_train, y_train)
        end = time.time()
    elif algorithm_name == 'NN':
        start = time.time()
        clf = MLPClassifier(random_state=1, max_iter=30000).fit(x_train, y_train)
        end = time.time()
    elif algorithm_name == 'LDA':
        start = time.time()
        clf = LinearDiscriminantAnalysis().fit(x_train, y_train)
        end = time.time()

    # predictions = clf.predict(x_test)
    predictions = cross_val_predict(clf, x_test, y_test, cv=10)
    metrics = get_metrics(y_test, predictions)
    plt.grid(None)

    return clf, (end - start), metrics

def save_model(model_name, model):
    extension = '.sav'
    directory = 'results/'
    file_name = directory + model_name + extension
    joblib.dump(model, file_name)

def adasyn_test(X, Y, algorithm='DT'):

    normal_timeouts = dict()
    normal_metrics = dict()

    inc_percentage = 0.05
    minority_percentage = 0.05
    last_balanced_accuracy = 0.0

    for i in range(0, ITERATIONS):
        print("ADASYN. Current minority percentage: " + str(minority_percentage))

        adsyn = ADASYN(minority_percentage, random_state=101)
        x_sm, y_sm = adsyn.fit_resample(X, Y)  # your imbalanced dataset is in X,y

        x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.3, random_state=101, stratify=y_sm)

        sc_X = StandardScaler()
        x_train_scaled = sc_X.fit_transform(x_train)
        x_test_scaled = sc_X.transform(x_test)

        print("ADASYN finished Current minority percentage: " + str(minority_percentage) + " getting model results.")

        dtree_normal, duration, metrics = get_results_for_algorithm(algorithm, x_train_scaled, y_train, x_test_scaled,
                                                                    y_test)

        if metrics[3] > last_balanced_accuracy:
            last_balanced_accuracy = metrics[3]
            save_model("ADASYN_MODELS/adasyn_optimal_" + str(algorithm), dtree_normal)
            print("ADASYN model stored with " + str(minority_percentage) + " sampled.")

        normal_timeouts[str(minority_percentage)] = duration
        normal_metrics[str(minority_percentage)] = metrics

        minority_percentage += inc_percentage
        minority_percentage = round(minority_percentage, 2)

        del x_sm
        del y_sm
        del x_train
        del x_test
        del y_train
        del y_test
        del x_train_scaled
        del x_test_scaled

    results_df = results_dataframe_from_dict(normal_metrics)
    results_df.to_csv("results/ADASYN_Metrics_" + algorithm + ".csv", index=False)

def borderline_smote_test(X, Y, algorithm='DT'):

    normal_timeouts = dict()
    normal_metrics = dict()

    inc_percentage = 0.05
    minority_percentage = 0.05
    last_balanced_accuracy = 0.0

    for i in range(0, ITERATIONS):
        print("BORDERLINE Current minority percentage: " + str(minority_percentage))

        sm = BorderlineSMOTE(minority_percentage, random_state=101)
        x_sm, y_sm = sm.fit_resample(X, Y)  # your imbalanced dataset is in X,y

        x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.3, random_state=101, stratify=y_sm)

        sc_X = StandardScaler()
        x_train_scaled = sc_X.fit_transform(x_train)
        x_test_scaled = sc_X.transform(x_test)

        print("BorderlineSMOTE finished Current minority percentage: " + str(minority_percentage) + " getting model results.")

        dtree_normal, duration, metrics = get_results_for_algorithm(algorithm, x_train_scaled, y_train, x_test_scaled,
                                                                    y_test)

        if metrics[3] > last_balanced_accuracy:
            last_balanced_accuracy = metrics[3]
            save_model("BORDERLINE_SMOTE_MODELS/borderlinesmote_optimal_" + str(algorithm), dtree_normal)
            print("Borderline model stored with " + str(minority_percentage) + " sampled.")

        normal_timeouts[str(minority_percentage)] = duration
        normal_metrics[str(minority_percentage)] = metrics

        minority_percentage += inc_percentage
        minority_percentage = round(minority_percentage, 2)

        del x_sm
        del y_sm
        del x_train
        del x_test
        del y_train
        del y_test
        del x_train_scaled
        del x_test_scaled

    results_df = results_dataframe_from_dict(normal_metrics)
    results_df.to_csv("results/BorderlineSMOTE_Metrics_" + algorithm + ".csv", index=False)
    
def smote_test(X, Y, algorithm='NN'):

    normal_timeouts = dict()
    normal_metrics = dict()

    inc_percentage = 0.05
    minority_percentage = 0.05
    last_balanced_accuracy = 0.0

    for i in range(0, ITERATIONS):
        print("SMOTE Current minority percentage: " + str(minority_percentage))

        over_smote = SMOTE(sampling_strategy=minority_percentage)
        x_sm, y_sm = over_smote.fit_resample(X, Y)

        x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.3, random_state=101, stratify=y_sm)

        sc_X = StandardScaler()
        x_train_scaled = sc_X.fit_transform(x_train)
        x_test_scaled = sc_X.transform(x_test)

        print("SMOTE finished Current minority percentage: " + str(minority_percentage) + " getting model results.")

        model, duration, metrics = get_results_for_algorithm(algorithm, x_train_scaled, y_train, x_test_scaled,
                                                                    y_test)

        if metrics[3] > last_balanced_accuracy:
            last_balanced_accuracy = metrics[3]
            save_model("SMOTE_MODELS/smote_optimal_" + str(algorithm), model)
            print("SMOTE model stored with " + str(minority_percentage) + " sampled.")

        normal_timeouts[str(minority_percentage)] = duration
        normal_metrics[str(minority_percentage)] = metrics

        minority_percentage += inc_percentage
        minority_percentage = round(minority_percentage, 2)

        del x_sm
        del y_sm
        del x_train
        del x_test
        del y_train
        del y_test
        del x_train_scaled
        del x_test_scaled

    results_df = results_dataframe_from_dict(normal_metrics)
    results_df.to_csv("results/SMOTE_Metrics_" + algorithm + ".csv", index=False)
    #return normal_metrics

def results_dataframe_from_dict(normal_metrics):
    final = pd.DataFrame(columns=["Minority %", "Metric", "Result"])
    for key in normal_metrics:
        final = final.append({'Minority %': key,
                              'Metric': 'Accuracy',
                              'Result': normal_metrics[key][0]},
                             ignore_index=True)

        final = final.append({'Minority %': key,
                              'Metric': 'Balanced Accuracy',
                              'Result': normal_metrics[key][3]},
                             ignore_index=True)

        final = final.append({'Minority %': key,
                              'Metric': 'Specificity',
                              'Result': normal_metrics[key][2]},
                             ignore_index=True
                             )

        final = final.append({'Minority %': key,
                              'Metric': 'Sensitivity',
                              'Result': normal_metrics[key][1]},
                             ignore_index=True
                             )
    return final

def encode_columns(dataset, X):
    i = 0
    for k in dataset.keys():
        if is_string_dtype(dataset[k]) == True:
            print(k)
            labelencoder_X = None
            labelencoder_X = LabelEncoder()
            X[:, i] = labelencoder_X.fit_transform(X[:, i])

        i = i + 1

if __name__ == '__main__':
    df = pd.read_csv('tumours_dataset.csv', low_memory=False)
    title = df.columns.tolist()[0:len(df.columns)-1]
    
    # Remove the cases where the year of the first diagnoses is 2016 and some columns
    is_not_2016 = df['YearFirstDiagnoses']!=2016
    df = df[is_not_2016]
    df = df.drop(columns=['SecondaryLocation', 'SubsequentsDesc', 'SmokingAfterFirstCancer', 'YearFirstDiagnoses', 'ObservedYears', 'AgeGroup', 'PrimaryLocation','BMIGroup'])

    #  ['DT', 'RF', 'BNB', 'GNB', 'NN', 'LR', 'LDA', 'SVM']
    models = ['DT', 'RF', 'BNB', 'GNB', 'NN', 'LR', 'LDA', 'SVM'] 
	
    print(df)
    # Encoding the categorical data
    X = df.iloc[: , 0:len(df.columns)-2].values
    Y = df.iloc[: , len(df.columns)-1].values

    encode_columns(df, X)
    print(X)
    print(Y)

    
    for model in models:
        smote_test(X, Y, model)
        borderline_smote_test(X, Y, model)
