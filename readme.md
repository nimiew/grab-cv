# How to reproduce experiment

1.	After cloning the repository, together with the scripts, folder should also contain "car_devkit.tgz", "cars_test.tgz", "cars_test_annos_withlabels.mat", "cars_train.tgz" downloaded from https://ai.stanford.edu/~jkrause/cars/car_dataset.html \
	On the command line, run "pip install -r requirements.txt"

2.	On the command line, run "python preprocess.py" *Can take quite long* \n
	This script splits the data into the respective train, validation, test folders in the data folder. It also parses "cars_test_annos_withlabels.mat" to create "test_labels.txt" for testing later \
	Each partition will be used for a certain purpose. Train data will be used to fit the model. Validation data is used for tuning parameters and trying different models. Testing data is finally used to estimate generalization error on future unseen data
	
3.	On the command line, run "python car_train.py" \
	You can choose which gpu to use by changing "os.environ["CUDA_VISIBLE_DEVICES"]" in line 3. \
	This script trains a model on the training data using transfer learning and finetuning. Due to computational constraints, I used EfficientNetB3 which has state-of-the-art performance for its model size (in terms of number of parameters). This is a new model from Google Brain, details can be found here - https://arxiv.org/pdf/1905.1194.pdf \
	The model trained will be saved as "model.h5"

4.	Rename "model.h5" as "model_final.h5"

5.	On the command line, run "python predict.py" \
	This script will produce:
	1.	predictions - "pred_labels.txt"
	2.	predictions with probabilities - "pred_labels_with_probabilities.txt" \
	NOTE: This may take some time

6.	On the command line, run "evaluate.py" \
	This script calculates accuracy based on "pred_labels_with_probabilities.txt" and "test_labels.txt" \
	The model uploaded on github has an accuracy of 0.8627036438253949
	
# How to test on other data
1.	Put custom test images in the folder data\custom_test\images \
	Place custom test labels (ground truth) in a custom_test.txt. Follow format of Stanford Cars dataset (in devkit\train_perfect_preds.txt)

2.	On the command line, run "python predict.py --custom_test" *Remember to run with the flag* \
	This script will produce:
	1.	predictions - "custom_pred_labels.txt"
	2.	predictions with probabilities - "custom_pred_labels_with_probabilities.txt" \
	NOTE: This may take some time

3.	On the command line, run "python evaluate.py --custom_test" *Remember to run with the flag* \
	This script calculates accuracy based on "custom_pred_labels_with_probabilities.txt" and "custom_test_labels.txt" 

# Other Notes:
1.	Run on Python 3
2.	Code tested on Linux with conda virtual environment
