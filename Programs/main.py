from load_and_preprocess_BES import *
from processing_for_modelling import * 
from implementing_lr import *
from implemeting_rf import *
from implemting_gb import * 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")

##############MAIN SCRIPT#########
#TThis script calls functions from other modules to clean and preprocess BES data. It then builds and evaluates three machine learning models: Logistic Regression, Random Forest, and Gradient Boosting.
## Full descripion of functions can be seen in indivdual scripts

#FILE PATHS
BES_2019 = "C:\\Users\\Hamid\\OneDrive\\Dissertation Code\\Output\\F2F_2019.xlsx"
#1) Load and export BES data into excel files.
load_and_export_BES_data()
#2 - Preprocess the 2019 BES dataset by selecting relevant columns, handling missing values, and renaming columns for clarity.
F2F_2019_df = preprocess_BES(BES_2019)
#continue cleaning and preprocssing data frame, 
df_encoded = preprocess_and_encode_variables(F2F_2019_df)

#3 Model Implementation
#train test split
train_df, test_df = create_train_test_split(df_encoded)
#LOGISTIC REGRESSION
#unbalanced
fit_multiclass_lr_cv(train_df, test_df)
#balanced
fit_multiclass_lr_balanced(train_df, test_df)
###RANDOM FOREST###
#balanced
fit_multiclass_rf_balanced(train_df, test_df)
###GRADIENT BOOSTING###
#balanced
fit_multiclass_gb_balanced(train_df, test_df)







