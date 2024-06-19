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
### Question 1: Can you visualize the data projected onto two principal components?
To visualize the data projected onto two principal components, we first need to perform PCA on the data.
Then, we can plot the data points in the 2D space of the first two principal components.
This will give you a visualization of the data projected onto the first two principal components.

### Question 2: How to determine the optimal number of principal components using pca.explained_variance_? Explain your selection process.
To determine the optimal number of principal components, we can look at the cumulative explained variance ratio. The explained variance ratio tells us how much of the total variance in the data is explained by each principal component.
The selection process is as follows:

1. Look at the plot of the cumulative explained variance ratio.
2. Determine the "elbow" or "knee" point in the plot, which is the point where the curve starts to flatten out.
3. The number of principal components corresponding to the elbow point is a good choice for the optimal number of components to use.

In this case, the plot shows that the cumulative explained variance ratio reaches around 95% with about 500 principal components. So a reasonable choice for the optimal number of principal components would be 500.

## 3. Image Classification:
### Data Preparation:
1. The original data is loaded, and the pixels column is converted to a numpy array.
2. The data is split into training and testing sets using train_test_split.
3. The data is scaled using StandardScaler.

### Hyperparameter Tuning:
1. Parameter grids are defined for each of the four classifiers: RandomForestClassifier, DecisionTreeClassifier, MLPClassifier, and KNeighborsClassifier.
2. The perform_grid_search function is used to perform a grid search for each classifier, finding the best set of hyperparameters and the corresponding cross-validation score.
3. The best estimators are then trained on the full training set.

### Model Evaluation:
The best estimators are tested on the held-out test set, and various performance metrics (accuracy, F1 score, recall, and precision) are calculated and stored in the metrics dictionary.

### PCA Transformation:
1. The original data is transformed using PCA, keeping 100 principal components to retain 90% of the information.
2. The transformed data is again split into training and testing sets, and scaled using StandardScaler.

### Hyperparameter Tuning with PCA:
The same process of hyperparameter tuning, training, and evaluation is repeated on the transformed data, and the results are stored in the metrics_pca dictionary.



















