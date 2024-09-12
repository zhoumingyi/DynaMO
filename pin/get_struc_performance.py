def process_file(input_file):
    op_names = []
    labels = []
    predictions = []

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.split('.')
            op_name = parts[0].strip()
            label_type = parts[1].strip()

            op_names.append(op_name)
            if label_type == "ObfOptions":
                labels.append(1)
                predictions.append(0)
            else:
                labels.append(0)
                predictions.append(0)

    return op_names, labels, predictions

def get_predictions(prediction_file, op_names, predictions):
    pred_name = []

    with open(prediction_file, 'r') as file:
        for line in file:
            op_name = line.strip()
            pred_name.append(op_name)
    
    for i, op_name in enumerate(op_names):
            if op_name in pred_name:
                predictions[i] = 0
            else:
                predictions[i] = 1

    return predictions

def calculate_metrics(labels, predictions):
    assert len(labels) == len(predictions), "Labels and predictions lists must have the same length."

    true_positive = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
    true_negative = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
    false_positive = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
    false_negative = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)

    accuracy = (true_positive + true_negative) / len(labels)
    fpr = false_positive / (false_positive + true_negative)
    fnr = false_negative / (false_negative + true_positive)

    return accuracy, fpr, fnr

# Define the input file paths
input_file = '../oplist.txt'
prediction_file = 'real_opnames.csv'

# Process the input file to get the operation names and labels
op_names, labels, predictions = process_file(input_file)

# Get the predictions based on the prediction file
predictions = get_predictions(prediction_file, op_names, predictions)

# Calculate the accuracy, FPR, and FNR
accuracy, fpr, fnr = calculate_metrics(labels, predictions)

# Print the results
print("Operation Names:", op_names)
print("Labels:", labels)
print("Predictions:", predictions)
print(f"Accuracy: {accuracy * 100.0:.2f}")
print(f"False Positive Rate (FPR): {fpr * 100:.2f}")
print(f"False Negative Rate (FNR): {fnr * 100:.2f}")
