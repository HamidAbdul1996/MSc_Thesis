import os
from pathlib import Path
import pandas as pd

def load_and_export_BES_data():
    """
Script to load British Election Study (BES) face-to-face survey dataset (2019),
process them into pandas DataFrames, and export them as Excel files for further analysis.
    """

    print("Starting the BES data loading and exporting process...")

    # Set working directory
    base_path = Path("C:/Users/Hamid/OneDrive/Dissertation Code")
    print(f"Setting working directory to: {base_path}")
    os.chdir(base_path)

    print("Loading 2019 BES dataset...")
    F2F_2019 = pd.read_stata('Input/Face to Face Survey/bes_rps_2019_1.3.0.dta', convert_categoricals=False)
    print("2019 BES dataset loaded successfully.")

    # Export to Excel
    print("Exporting 2019 BES dataset to Excel...")
    output_path_2019 = base_path / "Output" / "F2F_2019.xlsx"
    F2F_2019.to_excel(output_path_2019, index=False)
    print(f"2019 BES dataset exported to: {output_path_2019}")

    print("All datasets loaded and exported successfully.")




def preprocess_BES(file_path):
    
    """
Function to preprocess the 2019 BES dataset. This function:
1. Loads the dataset from the specified Excel file.
2. Selects relevant columns for analysis.
3. Fills missing values in the 'Voting Outcome' column with a placeholder for non-voters.
4. Renames columns for clarity and better interpretability.
5. Saves the preprocessed DataFrame as 'F2F_2019_preprocessed.xlsx' in the same directory as the input file.
    """


    print(f"Starting preprocessing of the 2019 BES dataset from: {file_path}")
    
    #Load the dataset 
    print("Loading the 2019 BES dataset from Excel...")
    F2F_2019 = pd.read_excel(file_path)
    print("Dataset loaded successfully.")
    
    # Select relevant columns for analysis
    print("Selecting relevant columns from the dataset...")
    F2F_2019_df = F2F_2019[[
        "b02", "y01_Annual", "y03", "y06b", "y09", "y10_banded",
        "y11", "edlevel", "y17", "region", "country", "Constit_Code", "Constit_Name", "LA_UA_Code" #"wt_vote"
    ]]
    print("Columns selected successfully.")
    
    # Handle missing values in the 'b02' (Voting Outcome) column
    print("Filling missing values in the 'b02' column (Voting Outcome)...")
    F2F_2019_df['b02'] = F2F_2019_df['b02'].fillna(13)  # 13 represents non-voters
    print("'b02' column missing values filled with 13 (non-voter).")

    #Drop rows with missing values (Only education level has missing)
    F2F_2019_df.dropna(inplace=True)
    
    #Rename columns for clarity
    print("Renaming columns for better interpretability...")
    F2F_2019_df = F2F_2019_df.rename(columns={
        'b02': 'b02 (Voting Outcome)',
        'y01_Annual': 'y01 (Income)',
        'y03': 'y03 (Homeownership)',
        'y06b': 'y06 (Religion)',
        'y09': 'y09 (Gender)',
        'y10_banded': 'y10 (Age)',
        'y11': 'y11 (Ethnicity)',
        'edlevel': 'y13 (Education)',
        'y17': 'y17 (Employment Status)'
    })
    
    #clean column names
    print("Step 1: Cleaning column names...")
    F2F_2019_df.columns = (F2F_2019_df.columns
                      .str.replace(' ', '_')    # Replace spaces with underscores
                      .str.replace('(', '')     # Remove opening parentheses
                      .str.replace(')', '')     # Remove closing parentheses
                      .str.replace('-', '_'))
    

    #Save the filtered DataFrame as 'F2F_2019_filtered.xlsx'
    output_path = Path(file_path).parent / "F2F_2019_filtered.xlsx"
    print(f"Saving the preprocessed dataset to: {output_path}")
    F2F_2019_df.to_excel(output_path, index=False)
    print("Preprocessed dataset saved successfully.")

    print("\nMissing values in each column:")
    print(F2F_2019_df.isnull().sum())
    
    return F2F_2019_df
