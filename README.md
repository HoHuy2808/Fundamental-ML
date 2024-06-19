# Mini-Project for Fundamentals of Machine Learning Course
![background](./materials/ai_wp.jpg)
This repository contains the code and data for a mini-project on facial expression recognition using machine learning algorithms.

## 📑 Project Policy
- Team: group should consist of 3-4 students.

    |No.| Student Name    | Student ID |
    | --------| -------- | ------- |
    |1|Hồ Huy|21110307|
    |2|Võ Hoàng Khang|21110317|
    |3|Lê Huỳnh Minh Tâm|21110172|
    |4|||

- The submission deadline is strict: **11:59 PM** on **June 22nd, 2024**. Commits pushed after this deadline will not be considered.

## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. **Fork the Project**: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. **Download the Dataset**: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the **/data** directory:

3. **Complete the Tasks**: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    2. Principle Component Analysis
    3. Image Classification
    4. Evaluating Classification Performance 

    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.

5. **Commit and Push Your Changes**: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push them to your forked repository on GitHub.


Feel free to modify and extend the notebook to explore further aspects of the data and experiment with different algorithms. Good luck.

# **Mini-Project for Fundamentals of Machine Learning Course:
## Motivation:
One motivation for representation learning is that learning algorithms can design features more effectively and efficiently than humans can. However, this challenge does not explicitly require entries to use representation learning. The dataset, assembled from the internet, is designed for facial expression classification.

## Dataset:
The data consists of grayscale images of faces, each measuring 48x48 pixels. The faces have been automatically aligned to be roughly centered and occupy a similar area within each image. The task is to categorize each face based on the emotion expressed, assigning it to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
The dataset contains a total of 35,887 examples.

## 1. Prerequisites:
The following Python libraries are required:
1. pandas
2. numpy
3. cv2
4. os
5. matplotlib.pyplot
The parse_data() and show_img() functions are provided to read the dataset and visualize the images and labels.

### Data Analysis:
The dataset is analyzed to understand the distribution of emotion occurrences. The results show that the mean of all emotion counts is {m}.

### Usage:
To run the facial expression recognition model, follow these steps:
1. Load the dataset:
2. Parse the data:
3. Visualize sample images and labels:

## 2. Principle Component Analysis
### Overview:
This code snippet demonstrates the use of PCA for dimensionality reduction and visualization of data projected onto principal components. It also explains how to determine the optimal number of principal components using explained variance.

### Libraries Used:
1. `scikit-learn` for PCA implementation (PCA class).
2. `numpy` for numerical operations.
3. `matplotlib.pyplot` for data visualization.

### Steps:
#### Visualization of Data Projected onto Two Principal Components
1. The data (`images`) is reshaped as necessary for PCA.
2. PCA is applied with `n_components=2` to project the data onto two principal components.
3. The transformed data (`images_pca`) is visualized using a scatter plot, where each point represents a sample in the dataset projected onto the two principal components.

#### Determining the Optimal Number of Principal Components:
1. PCA is performed again with a higher number of components (`n_components=2`).
2. The cumulative explained variance ratio (`cumulative_sum`) is plotted against the number of components to visualize how much variance is explained as the number of components increases.
3. The plot help in determining the optimal number of principal components based on the cumulative explained variance.

### Explanation:
1. Visualization: PCA reduces the dimensionality of the data while preserving as much variance as possible. Visualizing data projected onto two principal components provides insight into the distribution and separation of data points in reduced dimensions.

2. Optimal Number of Components: The plot of cumulative explained variance helps in deciding how many principal components to retain. The elbow or knee point in the plot, where adding more components doesn't significantly increase the explained variance, indicates a reasonable choice for the number of principal components.

### Usage:
1. Ensure `scikit-learn`, `numpy`, and `matplotlib` are installed.
2. Replace `images` with your dataset or data array.
3. Adjust `n_components` based on the number of principal components desired for analysis.
4. Run the script to visualize data in reduced dimensions and determine the optimal number of principal components.

### Notes:
1. PCA is useful for reducing data dimensionality, especially in cases where the original dataset has a large number of features.
2. The optimal number of principal components should balance computational efficiency with the amount of variance explained by the reduced dimensions.



## 3. Image Classification:
### Overview:
This code snippet demonstrates how to perform image classification using different machine learning classifiers both with and without Principal Component Analysis (PCA). The goal is to evaluate the performance of these classifiers in terms of accuracy, F1 score, recall, and precision on a dataset of images represented as pixel values.

### Libraries Used:
`scikit-learn` for building classifiers, preprocessing data, and evaluating performance.
`numpy` for handling array operations efficiently.

### Steps:
#### Original Data:
1. The original dataset (df) contains emotion labels (y) and pixel values (X). The pixel values are processed from strings to numpy arrays.
2. Data is split into training and test sets using `train_test_split`.

#### Classifiers without PCA:
1. Classifiers such as RandomForestClassifier, DecisionTreeClassifier, MLPClassifier, and KNeighborsClassifier are initialized with predefined parameter grids for hyperparameter tuning.
2. Grid search (GridSearchCV) is used to find the best hyperparameters for each classifier based on accuracy.
3. The best model for each classifier is trained on the training set and evaluated on the test set.
4. Metrics including accuracy, F1 score, recall, and precision are calculated and stored for further analysis.

#### Classifiers with PCA
1. PCA is applied to reduce the dimensionality of the data while retaining 90% of its variance. The number of principal components chosen is 100 based on a prior analysis.
2. The data is standardized using StandardScaler after PCA transformation.
3. The same set of classifiers and parameter grids are used as in the original data case.
4. Similarly, hyperparameter tuning, training, and evaluation are performed for each classifier.
5. Metrics (accuracy, F1 score, recall, precision) for classifiers with PCA are stored separately for comparison.

### Code Explanation:
The provided code includes functions and loops to automate:
1. Hyperparameter tuning (perform_grid_search function).
2. Training and evaluation of models.
3. Storage of performance metrics in dictionaries (metrics and metrics_pca).

### Usage:
1. Ensure scikit-learn, numpy, and relevant dependencies are installed.
2. Replace df with your dataset containing emotions and pixel values.
3. Adjust parameter grids (param_grids) and classifier settings based on specific requirements.
4. Run the script to obtain performance metrics and compare the effectiveness of classifiers with and without PCA.

### Notes:
1. Performance metrics such as accuracy, F1 score, recall, and precision are essential for evaluating classification models.
2. Results may vary based on dataset characteristics, parameter settings, and preprocessing steps.



## 4. Evaluating Classification Performance
### Overview:
This code snippet demonstrates how to evaluate the performance of classification models using various metrics such as accuracy, F1 score, recall, and precision. It compares models with and without Principal Component Analysis (PCA) and identifies the best-performing models based on different metrics.

### Code Explanation:
#### Best Model without PCA:
This part identifies and prints the best-performing model without PCA based on:
1. Model: The classifier achieving the highest accuracy.
2. Accuracy: Highest accuracy score among all classifiers.
3. F1 Score: Highest F1 score among all classifiers.
4. Recall: Highest recall score among all classifiers.
5. Precision: Highest precision score among all classifiers.

### Usage:
To use this code:
1. Ensure that metrics and metrics_pca dictionaries/lists are correctly populated with classifier names and corresponding evaluation metrics (accuracy, F1 score, recall, precision).
2. Run the code to determine and display the best models and their respective performance metrics.

### Notes:
The code assumes that metrics and metrics_pca are structured dictionaries/lists containing classifier names and their corresponding evaluation metrics.












