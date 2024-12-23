# Comparing Predictive Models for Electoral Behaviour in England and Wales: Logistic Regression vs Tree-Based Algorithms
## Overview

This project investigates the effectiveness of different predictive models in analysing electoral behaviour in the UK. It compares Logistic Regression, Random Forest, and Gradient Boosting algorithms to determine their suitability for predicting voting patterns based on demographic and socioeconomic variables.

## Key Objectives

Assess Model Performance: Evaluate the accuracy and reliability of each model in predicting voting behaviour.

Understand Demographic and Socioeconomic Influences: Analyse how demographic and socioeconomic factors shape electoral behaviour.

Methodological Improvements: Explore enhancements to traditional Multilevel Regression and Poststratification (MRP) frameworks.

## Data Source

The research utilises data from the British Election Study (BES) 2019 Face-to-Face survey, which provides detailed demographic and socioeconomic information along with voting behaviour data.

## Models Implemented

Logistic Regression: Serves as the baseline model, useful for interpretable relationships between demographics and voting patterns.

Random Forest: Captures non-linear relationships and interactions between predictors, balancing complexity and interpretability.

Gradient Boosting: Focuses on capturing subtle patterns in high-dimensional data, particularly in complex urban settings.

## Methodology

Preprocessing: Data cleaning, encoding categorical variables, and handling class imbalances through weighted sampling.

Feature Engineering: Incorporation of constituency-level aggregates to reflect local socioeconomic context.

Evaluation Metrics: Accuracy, Macro F1 Score, and feature importance analysis to compare models.

Validation: Used train-test splits at the constituency level to ensure robustness and generalisability.

## Key Findings

Logistic Regression performs well in stable, demographically homogeneous areas but struggles with diverse or evolving populations.

Random Forest balances complexity and interpretability, making it effective for marginal constituencies with moderate demographic changes.

Gradient Boosting excels in urban and high-density areas where traditional demographic predictors break down, capturing complex interactions.

## Repository Contents

/Input/ - Raw BES 2019 data

/Programs/ - Python scripts for data preprocessing, model training, and evaluation.

/Ouputs/ - In-depth results for all outputs for each model. Also visualsiation.

requirements.txt - Lists all dependencies required to run the project.

README.md - Project documentation (this file).

## Dependencies

Python 3.8 or higher


Installation

Clone the repository:
```
git clone https://github.com/username/repository-name.git
```
Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate    # Linux/MacOS
.\venv\Scripts\activate     # Windows
```
Install required libraries:
```
pip install -r requirements.txt
```
Execute the Jupyter notebooks or Python scripts for analysis.

## Usage

Preprocessing: Run scripts/preprocessing.py to clean and prepare data.

Model Training: Execute scripts/train_models.py to train and evaluate models.

Analysis: Use notebooks in the /notebooks/ folder to visualise results and insights.

## Future Work

Incorporate hierarchical models to address geographic variations more effectively.

Extend the analysis to include turnout predictions and tactical voting behaviours.

Apply hybrid approaches combining machine learning with traditional MRP frameworks.

## Author

Hamid Abdul,
University of Essex,
MSc Computing Project December 2024

## Acknowledgements

British Election Study (BES) for providing the data.

Supervisors and mentors for their guidance throughout the research process.

