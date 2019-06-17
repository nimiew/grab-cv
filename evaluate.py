import argparse
parser = argparse.ArgumentParser(description='produce predictions')
parser.add_argument('--custom_test', action='store_true', help='flag for testing custom data. refer to readme')
args = parser.parse_args()

from sklearn.metrics import accuracy_score
import os

if(args.custom_test):
    preds_file = open("custom_pred_labels_with_probabilities.txt", "r")
    truths_file = open("custom_test_labels.txt", "r")
else:
    preds_file = open("pred_labels_with_probabilities.txt", "r")
    truths_file = open("test_labels.txt", "r")

pred_labels = []
pred_probs = []
for pred_line in preds_file.readlines():
    pred_labels.append(int(pred_line[:-1].split(" ")[0]))
    pred_probs.append(float(pred_line[:-1].split(" ")[1]))

test_labels = [int(line[:-1]) for line in truths_file.readlines()]

print(accuracy_score(test_labels, pred_labels))

