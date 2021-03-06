import sys
import statistics
import pandas
import numpy
from math import e
from statistics import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

FILENAME = "Project1Data.csv"
POSSIBLE_ARGS = ["dia", "sys", "eda", "res", "all"]
ARG_FULL_NAMES = ["BP Dia_mmHg", "LA Systolic BP_mmHg", "EDA_microsiemens", "Respiration Rate_BPM"]

def format_data(csv_data, command_arg):
    columns = ['ID', 'Type', 'Class', 'Data']
    csv_data[columns] = csv_data.Data.str.split(',', n=3, expand=True)
    csv_data.Data = csv_data.Data.str.split(',')
    if command_arg != "all":
        type_full_name = ARG_FULL_NAMES[POSSIBLE_ARGS.index(command_arg)]
        csv_data = csv_data[csv_data['Type'] == type_full_name]
    return csv_data

# mean, variance, entropy, min, and max
def specific_feature_calculation(csv_data):
    features = []
    for _, row in csv_data.iterrows():
        data_list = [float(val) for val in row['Data']]
        mean = statistics.mean(data_list)
        variance = statistics.variance(data_list)
        value_counts = pandas.Series(data_list).value_counts(normalize=True, sort=False)
        entropy = -(value_counts * numpy.log(value_counts)/numpy.log(e)).sum()
        features.append([mean, variance, entropy, min(data_list), max(data_list)])
    return features

def all_feature_calculation(csv_data):
    features = []
    for type_name in ARG_FULL_NAMES:
        print("Calculating " + type_name + " features")
        csv_data_typed = csv_data[csv_data['Type'] == type_name]
        calculations = specific_feature_calculation(csv_data_typed)
        if not features:
            features = calculations
        else:
            [feat.extend(calc) for feat, calc in zip(features, calculations)]
    data_target = [pain['Class'] == "Pain" for _, pain in csv_data_typed.iterrows()]
    return features, data_target

# confusion matrix, classification accuracy, precision, and recall.
def random_forest_predictions(calculations, targets):
    pain_forest = RandomForestClassifier()
    accuracy_list, precision_list, matrix_list, recall_list = [], [], [], []
    kfold = model_selection.KFold(n_splits = 10)
    np_calc = numpy.asarray(calculations)
    np_targets = numpy.asarray(targets)

    for train_i, test_i in kfold.split(calculations):
        pain_forest.fit(np_calc[train_i], np_targets[train_i])
        target_pred = pain_forest.predict(np_calc[test_i])
        accuracy_list.append(accuracy_score(np_targets[test_i], target_pred))
        precision_list.append(precision_score(np_targets[test_i], target_pred))
        recall_list.append(recall_score(np_targets[test_i], target_pred))
        matrix_list.append(numpy.matrix(confusion_matrix(np_targets[test_i], target_pred)))

    print("\nAccuracy avg:\t" + str(mean(accuracy_list)))
    print("Precision avg:\t" + str(mean(precision_list)))
    print("Recall avg: \t" + str(mean(recall_list)))
    print("Confusion matrix avg:")
    print(sum(matrix_list)/len(matrix_list))


if __name__ == "__main__":
    if len(sys.argv) != 2 or str(sys.argv[1]) not in POSSIBLE_ARGS:
        raise Exception("Requires one command line argument (\'dia\', \'sys\', \'eda\', \'res\', \'all\').")
    data = pandas.read_csv(FILENAME, delimiter='|', names = ['Data'])
    command_arg = str(sys.argv[1])

    formatted_data = format_data(data, command_arg)
    if command_arg != "all":
        print("Calculating " + ARG_FULL_NAMES[POSSIBLE_ARGS.index(str(sys.argv[1]))] + " features")
        data_target = [pain['Class'] == "Pain" for _, pain in formatted_data.iterrows()]
        calculations = specific_feature_calculation(formatted_data)
    else:
        calculations, data_target = all_feature_calculation(formatted_data)
    random_forest_predictions(calculations, data_target)
