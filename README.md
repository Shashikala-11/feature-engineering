
# Feature Engineering & Data Preprocessing

### A Practical Repository on Preparing, Transforming and Engineering Data for Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas" />
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white" />
</p>

<p align="center">
  <b>Data Cleaning • Missing Value Imputation • Outlier Treatment • Feature Transformation • Encoding • Feature Construction • ML Pipelines</b>
</p>

---

## 📖 Overview

Feature Engineering is a fundamental component of the Machine Learning lifecycle. The quality, representation and preparation of input features directly influence how effectively a machine learning algorithm can identify patterns and generate predictions.

This repository is a collection of practical implementations, experiments and Jupyter notebooks focused on **Feature Engineering, Data Cleaning and Data Preprocessing** using Python and its machine learning ecosystem.

It explores techniques for handling incomplete datasets, identifying anomalies, transforming numerical distributions, encoding categorical variables, constructing meaningful features and preparing heterogeneous data for machine learning algorithms.

The repository also includes practical experiments involving the Titanic dataset and MNIST dataset to demonstrate preprocessing workflows, dimensionality reduction and model pipeline construction.

Rather than treating preprocessing as a collection of isolated operations, the focus is on understanding how individual transformations contribute to a structured and reproducible machine learning workflow.

---

## 🎯 Objectives

The primary objectives of this repository are to:

- Understand the importance of data quality in machine learning.
- Explore different strategies for handling missing observations.
- Identify and investigate outliers using statistical techniques.
- Apply appropriate scaling and normalization methods.
- Transform numerical features to improve their representation.
- Convert categorical variables into machine-readable formats.
- Construct meaningful features from existing variables.
- Understand dimensionality reduction using Principal Component Analysis.
- Work with heterogeneous numerical and categorical features.
- Implement preprocessing workflows using Scikit-learn Pipelines.
- Understand the difference between manual preprocessing and automated preprocessing pipelines.

---

## 📚 Topics & Implementations

### 1. Missing Value Handling

Missing observations can introduce bias, reduce available training data and affect model performance. This section explores different imputation strategies for handling incomplete datasets.

**Techniques explored:**

| Technique | Description |
|---|---|
| Mean Imputation | Replacing missing numerical values with the feature mean |
| Median Imputation | Using the median for numerical missing values |
| Mode Imputation | Replacing missing categorical values with the most frequent category |
| Constant Imputation | Filling missing values with a predefined constant |
| Random Sample Imputation | Replacing missing observations using randomly selected existing values |
| Missing Indicator | Creating additional features to identify originally missing observations |
| KNN Imputation | Estimating missing values using nearest neighboring observations |
| Multivariate Imputation | Predicting missing values using relationships among multiple features |
| MICE Algorithm | Iterative imputation using chained conditional models |

**Practical implementations:**

- [`handling missing data/`](./handling%20missing%20data/)
- [`KNN-Imputer-Multivariate-Imputation`](./KNN-Imputer-Multivariate-Imputation)
- [`MICE-Algorithm`](./MICE-Algorithm)
- [`numerical-data-univariate-simple-imputation.ipynb`](./numerical-data-univariate-simple-imputation.ipynb)
- [`missing-indicator-random-sample-imputation.ipynb`](./missing-indicator-random-sample-imputation.ipynb)
- [`categorical-data-handling-missing-data.ipynb`](./categorical-data-handling-missing-data.ipynb)
- [`mixed-variable-example.ipynb`](./mixed-variable-example.ipynb)

---

### 2. Outlier Detection & Treatment

Outliers are observations that differ significantly from the general distribution of a dataset. Depending on the problem, they may represent genuine extreme observations, measurement errors or unusual patterns.

**Techniques explored:**

| Technique | Description |
|---|---|
| Interquartile Range (IQR) | Identifying extreme values using quartiles |
| Statistical Outlier Detection | Investigating unusual observations through statistical distributions |
| Outlier Capping | Limiting extreme observations to predefined boundaries |
| Outlier Trimming | Removing observations outside selected thresholds |
| Distribution Visualization | Examining data distributions to identify potential anomalies |

**Practical implementations:**

- [`outliers/`](./outliers/)
- [`outlier_detection01.ipynb`](./outlier_detection01.ipynb)
- [`outlier_detection_using_IQR.ipynb`](./outlier_detection_using_IQR.ipynb)
- [`handling_outliers_example_01.ipynb`](./handling_outliers_example_01.ipynb)

---

### 3. Feature Scaling & Normalization

Features often have different numerical ranges and units. Scaling helps place numerical variables on comparable scales and is particularly important for distance-based and gradient-based algorithms.

**Techniques explored:**

| Technique | Description |
|---|---|
| Standardization | Transforming features using mean and standard deviation |
| Min-Max Scaling | Mapping values to a specified range |
| Robust Scaling | Scaling using median and interquartile range |
| Normalization | Scaling observations based on vector norms |

**Practical implementations:**

- [`feature_scaling_example.ipynb`](./feature_scaling_example.ipynb)
- [`normalization_example.ipynb`](./normalization_example.ipynb)

---

### 4. Feature Transformation

Feature transformation modifies the representation or distribution of numerical variables. It can help address skewness, improve numerical behavior and make certain patterns easier for algorithms to model.

**Techniques explored:**

- Log Transformation
- Power Transformation
- Box-Cox Transformation
- Yeo-Johnson Transformation
- Function-Based Transformation

**Practical implementations:**

- [`power-transformers-example.ipynb`](./power-transformers-example.ipynb)
- [`function_transformer_Example.ipynb`](./function_transformer_Example.ipynb)

---

### 5. Categorical Feature Encoding

Machine learning algorithms generally require numerical representations of input features. Categorical encoding converts categorical variables into suitable numerical formats.

**Techniques explored:**

| Technique | Description |
|---|---|
| One-Hot Encoding | Creating binary indicator columns for categories |
| Ordinal Encoding | Assigning numerical values based on category order |
| Binarization | Converting numerical values into binary representations |

**Practical implementations:**

- [`one_hot_encoding_example.ipynb`](./one_hot_encoding_example.ipynb)
- [`ordinal_encoding_example.ipynb`](./ordinal_encoding_example.ipynb)
- [`binarization_example.ipynb`](./binarization_example.ipynb)
- [`categorical-data-handling-missing-data.ipynb`](./categorical-data-handling-missing-data.ipynb)

---

### 6. Feature Construction & Discretization

Feature construction involves creating new variables from existing features to capture relationships that may not be directly represented in the original dataset.

Discretization converts continuous numerical values into discrete intervals or categories.

**Concepts explored:**

- Creating new features from existing variables.
- Extracting meaningful information from raw data.
- Converting continuous variables into categorical intervals.
- Understanding the effect of feature representation on machine learning.

**Practical implementations:**

- [`Feature-Construction.ipynb`](./Feature-Construction.ipynb)
- [`discretization_example.ipynb`](./discretization_example.ipynb)
- [`numerical_to_categorical_example.ipynb`](./numerical_to_categorical_example.ipynb)
- [`date-and-time-example.ipynb`](./date-and-time-example.ipynb)

---

### 7. Dimensionality Reduction – Principal Component Analysis

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms correlated features into a smaller set of uncorrelated components while retaining as much variance as possible.

**Concepts explored:**

- Dimensionality reduction.
- Feature correlation and variance.
- Principal component extraction.
- Explained variance.
- Representing high-dimensional data in lower-dimensional spaces.

**Practical implementation:**

- [`PCA-practical-example-on-mnist-dataset.ipynb`](./PCA-practical-example-on-mnist-dataset.ipynb)

**Dataset:** MNIST handwritten digit dataset.

---

### 8. Scikit-learn Transformers & ColumnTransformer

Real-world datasets frequently contain different feature types, requiring different preprocessing strategies.

Scikit-learn provides transformer utilities to organize and apply preprocessing operations systematically.

**Concepts explored:**

- ColumnTransformer
- FunctionTransformer
- Applying transformations to selected columns.
- Handling mixed numerical and categorical variables.
- Combining multiple preprocessing operations.

**Practical implementations:**

- [`column_transformer_example.ipynb`](./column_transformer_example.ipynb)
- [`function_transformer_Example.ipynb`](./function_transformer_Example.ipynb)
- [`mixed-variable-example.ipynb`](./mixed-variable-example.ipynb)

---

### 9. End-to-End Preprocessing & Machine Learning Pipelines

Preprocessing pipelines help organize multiple transformations into a consistent sequence of operations.

They reduce repetitive code, improve reproducibility and help prevent data leakage when fitted correctly.

**Concepts explored:**

- Manual preprocessing workflows.
- Automated preprocessing using Pipeline.
- Combining preprocessing and model training.
- Applying transformations consistently to training and testing data.
- Comparing pipeline-based and non-pipeline workflows.

**Practical implementations:**

- [`titanic_models/`](./titanic_models/)
- [`titanic_predict_using_pipeline.ipynb`](./titanic_predict_using_pipeline.ipynb)
- [`titanic_predict_without_pipeline.ipynb`](./titanic_predict_without_pipeline.ipynb)
- [`titanic_using_pipelining.ipynb`](./titanic_using_pipelining.ipynb)
- [`titanic_without_pipelining/`](./titanic_without_pipelining/)

**Dataset:** Titanic passenger dataset.

These experiments demonstrate how preprocessing operations can be integrated into a machine learning workflow before model training and prediction.

---

## 🗂️ Repository Structure

```text
feature-engineering/
│
├── dataset/
│
├── handling missing data/
│
├── outliers/
│
├── titanic_models/
│
├── Feature-Construction.ipynb
│
├── KNN-Imputer-Multivariate-Imputation/
│
├── MICE-Algorithm/
│
├── PCA-practical-example-on-mnist-dataset.ipynb
│
├── binarization_example.ipynb
├── categorical-data-handling-missing-data.ipynb
├── column_transformer_example.ipynb
├── date-and-time-example.ipynb
├── discretization_example.ipynb
├── feature_scaling_example.ipynb
├── function_transformer_Example.ipynb
├── handling_outliers_example_01.ipynb
├── missing-indicator-random-sample-imputation.ipynb
├── mixed-variable-example.ipynb
├── normalization_example.ipynb
├── numerical-data-univariate-simple-imputation.ipynb
├── numerical_to_categorical_example.ipynb
├── one_hot_encoding_example.ipynb
├── ordinal_encoding_example.ipynb
├── outlier_detection01.ipynb
├── outlier_detection_using_IQR.ipynb
├── power-transformers-example.ipynb
│
├── titanic_predict_using_pipeline.ipynb
├── titanic_predict_without_pipeline.ipynb
├── titanic_using_pipelining.ipynb
│
├── cars.csv
├── customer.csv
│
└── README.md
```

*Note: The structure above highlights the main directories, notebooks and datasets. Refer to the repository for the complete file listing.*

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations and array manipulation |
| Scikit-learn | Imputation, scaling, encoding, transformations and pipelines |
| Matplotlib | Data visualization |
| Jupyter Notebook | Interactive experimentation |
| SciPy | Statistical and scientific computing |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.x
- Jupyter Notebook or JupyterLab
- Basic knowledge of Python and statistics

### 1. Clone the repository

```bash
git clone https://github.com/Shashikala-11/feature-engineering.git
```

### 2. Navigate to the repository

```bash
cd feature-engineering
```

### 3. Create a virtual environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open any notebook to explore the corresponding implementation.

---

## 🔬 Practical Learning Outcomes

Working through these implementations provides exposure to several important aspects of the Machine Learning data preparation lifecycle:

| Area | Practical Understanding |
|---|---|
| Data Quality | Identifying incomplete, inconsistent and unusual observations |
| Missing Data | Selecting suitable imputation strategies |
| Statistical Analysis | Understanding distributions, quartiles and outliers |
| Feature Representation | Transforming numerical and categorical variables |
| Feature Engineering | Constructing meaningful features |
| Dimensionality Reduction | Understanding PCA and lower-dimensional representations |
| Preprocessing Automation | Using Scikit-learn transformers and pipelines |
| ML Workflow | Preparing data consistently before model training |

---

## ⚠️ Important Machine Learning Considerations

Preprocessing is not simply about applying every available transformation. The correct approach depends on the dataset, feature distributions and the learning algorithm.

Some important considerations include:

1. **Avoid Data Leakage:** Fit imputation, scaling and encoding transformations using training data only.
2. **Understand Missingness:** Choose imputation methods based on the data distribution and missing-value mechanism.
3. **Investigate Outliers:** Do not automatically remove every extreme observation; it may contain valuable information.
4. **Choose Appropriate Scaling:** Different algorithms respond differently to feature scales.
5. **Preserve Categorical Meaning:** Encoding methods should reflect the actual nature of categorical variables.
6. **Use Reproducible Pipelines:** Integrate preprocessing and model training where appropriate.
7. **Evaluate Transformations:** Compare the effect of preprocessing decisions instead of assuming that every transformation improves performance.

---

## 🚀 Future Scope

Potential areas for further exploration and expansion include:

- Advanced feature selection techniques.
- Recursive Feature Elimination (RFE).
- Mutual Information and statistical feature selection.
- Automated feature engineering.
- Advanced categorical encoding.
- Time-series feature engineering.
- Text preprocessing and NLP feature extraction.
- Computer vision preprocessing.
- Feature engineering with PySpark.
- Feature stores and production data pipelines.
- Experiment tracking and preprocessing reproducibility.

*These are planned areas of exploration, not claims about currently implemented notebooks.*

---

## 👩‍💻 Author

**Shashikala Gupta**

Computer Engineering | Data Science & Machine Learning

Interested in Machine Learning, Data Science, AI Engineering and building practical data-driven applications.

- **GitHub:** [Shashikala-11](https://github.com/Shashikala-11)
- **LinkedIn:** [Shashikala Gupta](https://www.linkedin.com/in/shashikala-gupta/)

---

## 📌 Repository

Explore the complete collection of notebooks, datasets and implementations:

**[Feature Engineering & Data Preprocessing – GitHub Repository](https://github.com/Shashikala-11/feature-engineering)**

If you find this repository useful, consider giving it a ⭐.

---

<p align="center">
  <b>Good Machine Learning Starts with Good Data.</b>
  <br/>
  Understanding the data is just as important as choosing the algorithm.
</p>
