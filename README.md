# mlmodelcstoolslab
its for a lab task in cs tools regarding a weak ml model that should get its accuracy increased upto by at least 2% .

Project Name: DecisionTreeCancerDetection

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset

This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The task is to predict whether the mass is benign or malignant.
Weak Model: Decision Tree Classifier

Decision trees are simple models that make predictions based on a series of decisions or rules. They are easy to interpret but tend to overfit the data if not pruned or regularized properly.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Breast Cancer Wisconsin (Diagnostic) Dataset

## Model Overview
This project aims to predict whether a breast mass is benign or malignant using the Breast Cancer Wisconsin (Diagnostic) Dataset. We are using a decision tree classifier as a weak machine learning model for this task.
-Initial or base reported accuracy is :
Accuracy of the decision tree classifier: 0.9473684210526315

## Model Specifications
- Model: Decision Tree Classifier
- Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
- Features: Computed from a digitized image of a fine needle aspirate (FNA) of a breast mass
- Task: Binary classification (benign or malignant)
- Evaluation Metric: Accuracy

## Initial Accuracy
The initial accuracy of the decision tree classifier on the test set is approximately 94.74%.

## Further Improvenent 
One popular ensemble method for decision trees is Random Forest, by using ensemblers , we 
have increased the accuracy of the model by 1.75%.


## Hyperparameters Tuned:

n_estimators: Number of trees in the forest.
max_depth: Maximum depth of the trees.
min_samples_split: Minimum number of samples required to split an internal node.
min_samples_leaf: Minimum number of samples required to be at a leaf node.

##Best Hyperparameters:

max_depth: 7
min_samples_leaf: 2
min_samples_split: 5
n_estimators: 100

## Accuracy of the Best Random Forest Classifier:

0.9649 (96.49%)

## Feature Scaling
result:Best hyperparameters: {'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
Accuracy of the best Random Forest classifier with feature scaling: 0.9649122807017544

conclusion:
Breast Cancer Wisconsin (Diagnostic) Dataset may not benefit significantly from feature scaling in the context of a Random Forest classifier.


## Project Structure
- `data`: Contains the dataset files.
- `code`: Contains the Python code for data preprocessing, model training, and evaluation.
- `results`: Contains the results of the model evaluation, including accuracy and any visualizations.

## Instructions
1. Clone the repository.
2. Install the required dependencies (`numpy`, `pandas`, `scikit-learn`, etc.).
3. Run the `train_model.py` script to train the model.
4. Run the `evaluate_model.py` script to evaluate the model and view the results.

## Future Work
- Hyperparameter tuning to improve model performance.
- Experimentation with other weak models and ensemble methods.
- Feature engineering to enhance model capabilities.

## Contributors
- [Ahsan Saleem (ME) ](https://github.com/icecoldkill)
- [Ahsan Saleem (ME 2) ](https://github.com/icecoldkill) //I did not find any other partner hence i will try to work and contribute via another git account.

## References
- [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

