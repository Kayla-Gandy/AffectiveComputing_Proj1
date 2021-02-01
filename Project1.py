import sys
import statistics
from math import e
from numpy import log
import pandas

FILENAME = "Project1Data.csv"
POSSIBLE_ARGS = ["dia", "sys", "eda", "res", "all"]
ARG_FULL_NAMES = ["BP Dia_mmHg", "LA Systolic BP_mmHg", "EDA_microsiemens", "Respiration Rate_BPM"]

def format_data(csv_data, command_arg):
    columns = ['ID', 'Type', 'Class', 'Data']
    csv_data[columns] = csv_data.Data.str.split(',', n=3, expand=True)
    csv_data.Data = csv_data.Data.str.split(',')
    if command_arg != "all":
        type_full_name = ARG_FULL_NAMES[POSSIBLE_ARGS.index(str(sys.argv[1]))]
        csv_data = csv_data[csv_data['Type'] == type_full_name]
    return csv_data
    # formatted = []
    # for index, row in csv_data.iterrows():
    #     pain = bool("No" not in row[2])
    #     formatted.append([pain, row[3]])
    # return formatted

# mean, variance, entropy, min, and max
def feature_calculation(csv_data):
    features = []
    for _, row in csv_data.iterrows():
        data_list = [float(val) for val in row['Data']]
        mean = statistics.mean(data_list)
        variance = statistics.variance(data_list)
        value_counts = pandas.Series(data_list).value_counts(normalize=True, sort=False)
        entropy = -(value_counts * log(value_counts)/log(e)).sum()
        features.append([mean, variance, entropy, min(data_list), max(data_list)])
    return features

if __name__ == "__main__":
    if len(sys.argv) != 2 or str(sys.argv[1]) not in POSSIBLE_ARGS:
        raise Exception("Requires one command line argument (\'dia\', \'sys\', \'eda\', \'res\', \'all\').")
    data = pandas.read_csv(FILENAME, delimiter='|', names = ['Data'])

    formatted_data = format_data(data, str(sys.argv[1]))
    calculations = feature_calculation(formatted_data)
    print(calculations)
