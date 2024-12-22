import pandas as pd
import numpy as np

def preprocess_and_encode_variables(df):
    """
    Preprocesses and encodes variables for models with:
    - Ordinal encoding for ordered variables
    - One-hot encoding for nominal variables with explicit reference categories
    - Proper handling of missing values
    """
    print("Starting preprocessing and encoding...")
    
    # Create a copy 
    df = df.copy()

    #Remove Scottish constituencies first
    df = df[~df['Constit_Code'].str.startswith('S14')].copy()
    print(f"Removed Scottish constituencies. Remaining constituencies: {len(df['Constit_Code'].unique())}")
    
    ###### 1. Handle ordinal variables#####
    
    ####EDUCATION####
    # Education with corrected ordering
    education_mapping = {
        0: 'No Qualifications',
        1: 'Below GCSE', 2: 'GCSE', 3: 'A-level', 
        4: 'Undergraduate', 5: 'Postgraduate'
    }
    education_order = ['No Qualifications', 'Below GCSE', 'GCSE', 'A-level', 'Undergraduate', 'Postgraduate']

    #Map numeric values to categories
    df['y13_Education'] = df['y13_Education'].map(education_mapping)
    df['y13_Education'] = pd.Categorical(df['y13_Education'], 
                                       categories=education_order, 
                                       ordered=True).codes
    
    ###INCOME###
    #mapping
    income_mapping = {
    1: 'Under £5,200',
    2: '£5,200 - £15,599',
    3: '£15,600 - £25,999',
    4: '£26,000 - £36,399',
    5: '£36,400 - £46,799',
    6: '£46,800 - £74,999',
    7: '£75,000 - £149,999',
    8: '£150,000 or more'
}
    #dropping rows where values are don't know/not stated/prefered not to say
    df = df[~df['y01_Income'].isin([-999, -2, -1])]

    income_order = [
    'Under £5,200',
    '£5,200 - £15,599',
    '£15,600 - £25,999',
    '£26,000 - £36,399',
    '£36,400 - £46,799',
    '£46,800 - £74,999',
    '£75,000 - £149,999',
    '£150,000 or more'
]
    # Map and convert to ordered categorical
    df['y01_Income'] = pd.Categorical(
    df['y01_Income'].map(income_mapping),
    categories=income_order,
    ordered=True
    ).codes


###AGE BANDING###
#mapping
    age_mapping = {
    1: '18-24',
    2: '25-34',
    3: '35-44',
    4: '45-54',
    5: '55-64',
    6: '65-74',
    7: '75-84',
    8: '85+'
}
    

# Define age order
    age_order = [
    '18-24',
    '25-34',
    '35-44',
    '45-54',
    '55-64',
    '65-74',
    '75-84',
    '85+'
]
    #dropping refusals
    df = df[~df['y10_Age'].isin([-2, -999])]


    # Map to categories and convert to ordered numeric
    df['y10_Age'] = df['y10_Age'].map(age_mapping)
    df['y10_Age'] = pd.Categorical(
        df['y10_Age'],
        categories=age_order,
        ordered=True
    ).codes
    
#2 ### Nominal Values ###

###EMPLOYMENT STATUS###
    
    # Employment Status
    employment_mapping = {
        1: 'Full-time', 2: 'Full-time',
        3: 'Part-time', 4: 'Part-time',
        5: 'Unemployed',
        6: 'Government sponsored training scheme', 
        7: 'Student',
        8: 'Homemaker',
        9: 'Sick_leave', 10: 'Sick_leave',
        11: 'Retired',
        12: 'Other'
    }

    #dropping refusals
    df = df[~df['y17_Employment_Status'].isin([-999])]

    df['y17_Employment_Status'] = df['y17_Employment_Status'].map(employment_mapping)

    # Then create dummy variables
    employment_dummies = pd.get_dummies(
        df['y17_Employment_Status'], 
        prefix='employment',
        drop_first=True 
    ).astype(int)

    # Add back to dataframe
    df = pd.concat([df, employment_dummies], axis=1)

    # Drop the original column
    df = df.drop('y17_Employment_Status', axis=1)

###HOME OWNERSHIP###
    home_ownership_mapping = {
        1: 'Own_home_outright',
        2: 'Own_home_on_mortgage',
        3: 'Rented',
        4: 'Rented',
        5: 'Belongs_to_Housing_Association',
        6: 'Other'
    }

    #dropping refusals
    df = df[~df['y03_Homeownership'].isin([-1, -999])]

    df['y03_Homeownership'] = df['y03_Homeownership'].map(home_ownership_mapping)

    # Then create dummy variables
    homeownership_dummies = pd.get_dummies(
        df['y03_Homeownership'], 
        prefix='homeownership',
        drop_first=True  
    ).astype(int)

    # Add back to dataframe
    df = pd.concat([df, homeownership_dummies], axis=1)

    # Drop the original column
    df = df.drop('y03_Homeownership', axis=1)


### Ethnicity ###
    ethnicity_mapping = {
        1: 'White', 2: 'White', 3: 'White', 4: 'White',
        5: 'Mixed', 6: 'Mixed', 7: 'Mixed', 8: 'Mixed',
        9: 'Asian', 10: 'Asian', 11: 'Asian', 12: 'Asian', 13: 'Asian',
        14: 'Black', 15: 'Black', 16: 'Black',
        17: 'Arab', 18: 'Other'
    }

     #dropping refusals
    df = df[~df['y11_Ethnicity'].isin([-1, 19])]

    df['y11_Ethnicity'] = df['y11_Ethnicity'].map(ethnicity_mapping)

    #Reorder categories to ensure 'White' is the baseline
    df['y11_Ethnicity'] = pd.Categorical(
    df['y11_Ethnicity'], 
    categories=['White','Mixed', 'Asian', 'Black', 'Arab', 'Other'],  # Explicit category order
    ordered=False
    )

       # Then create dummy variables
    ethnicity_dummies = pd.get_dummies(
        df['y11_Ethnicity'], 
        prefix='ethnicity',
        drop_first=True  
    ).astype(int)

    # Add back to dataframe
    df = pd.concat([df, ethnicity_dummies], axis=1)

    # Drop the original column
    df = df.drop('y11_Ethnicity', axis=1)

### Religion ###

    
    # Simplify religion categories
    religion_mapping = {
        0: 'No religion',
        1: 'Christian', 2: 'Christian', 3: 'Christian',
        4: 'Christian', 5: 'Christian', 6: 'Christian',
        7: 'Christian', 8: 'Christian', 9: 'Christian',
        10: 'Christian', 11: 'Christian', 12: 'Jewish',
        13: 'Hindu', 14: 'Muslim', 15: 'Sikh',
        16: 'Buddhist', 17: 'Other', 18: 'Christian',
        19: 'Christian', 20: 'Other', 21: 'Other', 22: 'Christian',
        23: 'Humanist', 24: 'Christian', 25: 'Hindu', 26: 'Christian',
        27: 'Other'
    }

    #dropping refusals
    df = df[~df['y06_Religion'].isin([-2])]

    # Map religion categories
    df['y06_Religion'] = df['y06_Religion'].map(religion_mapping)

    # Convert to categorical
    df['y06_Religion'] = pd.Categorical(
    df['y06_Religion'],
    categories=['Christian', 'No religion', 'Muslim', 'Hindu', 'Sikh', 'Buddhist', 'Other'],
    ordered=False
    )

          # Then create dummy variables
    religion_dummies = pd.get_dummies(
        df['y06_Religion'], 
        prefix='religion',
        drop_first=True  
    ).astype(int)

    # Add back to dataframe
    df = pd.concat([df, religion_dummies], axis=1)

    # Drop the original column
    df = df.drop('y06_Religion', axis=1)



 ###Gender###
     #  gender mapping
    gender_mapping = {
        1: 'Male',
        2: 'Female',
        3: 'In another way',
        4: 'Prefer not to say'
    }

    #dropping refusals and prefer not to say
    df = df[~df['y09_Gender'].isin([3, 4, -999])]

    # Map religion categories
    df['y09_Gender'] = df['y09_Gender'].map(gender_mapping)

    # Encode gender back to numeric values
    gender_encoding = {
    'Male': 0,
    'Female': 1
    }

    df['y09_Gender'] = df['y09_Gender'].map(gender_encoding)

    

#### Outcome Variable (Voting Outcome)#####
#    Simplify outcome categories
    party_map = {
    1: 'Labour',    # 
    2: 'Conservatives',    
    3: 'Liberal Democrats',   
    4: 'Scottish National Party',    
    5: 'Other',    
    6: 'Other',    
    7: 'Other',    
    8: 'Other',   
    9: 'Other',  
    10: 'Other',    
    11:'Other',
    12: 'Didnt Vote',
    13: 'Didnt Vote'    }

     #dropping refusals
    df = df[~df['b02_Voting_Outcome'].isin([-2, -1])]

    # Map the numeric values to party names for reference
    df['b02_Voting_Outcome'] = df['b02_Voting_Outcome'].map(party_map)

    # Encode party names back to numeric values
    party_encoding = {
    'Labour': 1,
    'Conservatives': 2,
    'Liberal Democrats': 3,
    'Scottish National Party': 4,
    'Other': 5,
    'Didnt Vote': 6
    }

    df['b02_Voting_Outcome'] = df['b02_Voting_Outcome'].map(party_encoding)
    
    
    print("\nPreprocessing completed.")
    print(f"Final number of features: {len(df.columns)}")
    
    # Print summary of transformations
    print("\nVariable types in processed dataset:")
    print(df.dtypes.value_counts())
    print("\nMissing values in processed dataset:")
    print(df.isnull().sum().sum())
    
    return df






