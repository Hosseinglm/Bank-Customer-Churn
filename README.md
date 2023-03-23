# Bank Customer Churn

This is a Python code for a machine learning task, specifically for handling an imbalanced dataset.
The task is to predict churn in a bank using a Support Vector Machine (SVM) algorithm.

The first part of the code is importing necessary libraries such as numpy, pandas, matplotlib_inline and seaborn.
Then, the dataset is loaded using pandas read_csv function, and the head and info of the dataset is printed. Next,
some exploratory data analysis (EDA) is performed on the dataset such as checking the duplicated records, null values,
and checking for the value count of categorical variables.

The next part of the code is handling the categorical variables using One-Hot Encoding.
The categorical variables are encoded using the get_dummies function from pandas. Then, the imbalance in the dataset is
handled using the Random Under Sampling (RUS) and Random Over Sampling (ROS) techniques from the Imbalanced-learn library.

The next step is splitting the data into training and testing sets.
There are three sets created: one from the original dataset, and two from RUS and ROS data.

Finally, feature scaling is applied using StandardScaler from scikit-learn library.
Support Vector Machine (SVM) algorithm is used to train the model on the training sets, and the model is tested on the testing sets.
The accuracy of the model is then measured using the classification_report and confusion_matrix functions from the scikit-learn library.

Overall, the code follows the standard machine learning pipeline: data preprocessing, EDA, handling imbalance data, train-test split,
feature scaling, and model training/testing. However, the comments and explanations in the code are sparse and incomplete.
It would be helpful if the author of the code provides more comprehensive comments and explanations of each step.
