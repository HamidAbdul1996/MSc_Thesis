from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def create_train_test_split(df, test_size=0.2, random_state=42):
    """
    Create train-test split while keeping constituencies together
    """
    # Get unique constituencies
    unique_constituencies = df['Constit_Code'].unique()
    
    # Split constituencies into train and test

    train_constituencies, test_constituencies = train_test_split(
        unique_constituencies, 
        test_size=test_size, 
        random_state=random_state
    )

    train_df = df[df['Constit_Code'].isin(train_constituencies)].copy()
    test_df = df[df['Constit_Code'].isin(test_constituencies)].copy()

    print("\nTrain-Test Split Summary:")
    print(f"Training set: {len(train_df)} records ({len(train_constituencies)} constituencies)")
    print(f"Test set: {len(test_df)} records ({len(test_constituencies)} constituencies)")

    return train_df, test_df


def fit_multiclass_lr_cv(train_df, test_df):
    """
    Fits a multi-class logistic regression model with constituency context
    for voting behavior prediction.
    """
    # Party mapping
    party_names = {
    1: 'Labour',    # 
    2: 'Conservatives',    
    3: 'Liberal Democrats',   
    4: 'Scottish National Party',    
    5: 'Other',    
    6: 'Didnt vote'
            }
    
    print("Fitting hierarchical model...")
    
    # Select predictors
    individual_predictors = [
        # Numeric/Ordinal variables
        'y01_Income', 'y09_Gender', 'y10_Age', 'y13_Education',
        
        # Employment dummies
        'employment_Homemaker', 'employment_Other', 'employment_Part-time',
        'employment_Retired', 'employment_Sick_leave', 'employment_Student',
        'employment_Unemployed',
        
        # Homeownership dummies
        'homeownership_Other', 'homeownership_Own_home_on_mortgage',
        'homeownership_Own_home_outright', 'homeownership_Rented',
        
        # Ethnicity dummies
        'ethnicity_Mixed', 'ethnicity_Asian', 'ethnicity_Black',
        'ethnicity_Arab', 'ethnicity_Other',
        
        # Religion dummies
        'religion_No religion', 'religion_Muslim', 'religion_Hindu',
        'religion_Sikh', 'religion_Buddhist', 'religion_Other'
    ]
    
    
    # Create clean copies of variables
    X_train = train_df[individual_predictors].copy()
    y_train = train_df['b02_Voting_Outcome'].copy()
    train_constituencies = train_df['Constit_Code'].copy()
    
    
    X_test = test_df[individual_predictors].copy()
    y_test = test_df['b02_Voting_Outcome'].copy()
    test_constituencies = test_df['Constit_Code'].copy()
  
    
    # Print initial class distribution
    print("\nClass distribution in training set:")
    train_dist = pd.Series(y_train).value_counts().sort_index()
    for party_id, count in train_dist.items():
        print(f"{party_names[party_id]}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Scale numeric variables
    numeric_vars = ['y13_Education', 'y01_Income', 'y10_Age']
    scaler = StandardScaler()
    X_train[numeric_vars] = scaler.fit_transform(X_train[numeric_vars])
    X_test[numeric_vars] = scaler.transform(X_test[numeric_vars])
    
    # Add constituency-level features
    train_constituency_stats = []
    for var in numeric_vars: 
        agg_stats = train_df.groupby('Constit_Code').apply(
            lambda x: np.average(x[var])
        ).reset_index()
        agg_stats.columns = ['Constit_Code', f'constituency_mean_{var}']
        train_constituency_stats.append(agg_stats)
    
    train_constituency_features = train_constituency_stats[0]
    for stats in train_constituency_stats[1:]:
        train_constituency_features = train_constituency_features.merge(
            stats, on='Constit_Code', how='outer'
        )
    
    # Same for test data
    test_constituency_stats = []
    for var in numeric_vars:
        agg_stats = test_df.groupby('Constit_Code').apply(
            lambda x: np.average(x[var])
        ).reset_index()
        agg_stats.columns = ['Constit_Code', f'constituency_mean_{var}']
        test_constituency_stats.append(agg_stats)
    
    test_constituency_features = test_constituency_stats[0]
    for stats in test_constituency_stats[1:]:
        test_constituency_features = test_constituency_features.merge(
            stats, on='Constit_Code', how='outer'
        )
    
    # Merge features
    X_train['Constit_Code'] = train_constituencies
    X_train = X_train.merge(train_constituency_features, on='Constit_Code', how='left')
    X_train = X_train.drop('Constit_Code', axis=1)
    
    X_test['Constit_Code'] = test_constituencies
    X_test = X_test.merge(test_constituency_features, on='Constit_Code', how='left')
    X_test = X_test.drop('Constit_Code', axis=1)

    # Define the parameter space
    param_distributions = {
        'C': loguniform(1e-3, 1e3),  
        'class_weight': ['balanced', None],
        'max_iter': [1000, 2000]
    }

    # Create base model
    base_model = LogisticRegression(solver='lbfgs')

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,  
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,  
        random_state=42,
        verbose=2
    )

    # Fit the random search
    print("Performing randomized search for optimal parameters...")
    random_search.fit(X_train, y_train)
    
    # Print results
    print("\nBest parameters found:")
    print(random_search.best_params_)
    print(f"\nBest cross-validation accuracy: {random_search.best_score_:.3f}")

    # Use the best model for predictions
    model = random_search.best_estimator_
    
    
    #Generate predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nOverall Accuracies:")
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")

    
    # Create results DataFrames
    train_results = pd.DataFrame({
        'Constit_Code': train_constituencies,
        'Actual_Vote': y_train,
        'Predicted_Vote': train_pred,
        'Age': train_df['y10_Age'],
        'Gender': train_df['y09_Gender'],
        'Education': train_df['y13_Education'],
        'Income': train_df['y01_Income'],
        'Region': train_df['region'] if 'region' in train_df.columns else None

    })
    
    test_results = pd.DataFrame({
        'Constit_Code': test_constituencies,
        'Actual_Vote': y_test,
        'Predicted_Vote': test_pred,
        'Age': test_df['y10_Age'],
        'Gender': test_df['y09_Gender'],
        'Education': test_df['y13_Education'],
        'Income': test_df['y01_Income'],
        'Region': test_df['region'] if 'region' in test_df.columns else None
        
    })
    
    # Add probability columns with party names
    for i, party in enumerate(model.classes_):
        train_results[f'Prob_{party_names[party]}'] = train_probs[:, i]
        test_results[f'Prob_{party_names[party]}'] = test_probs[:, i]
    
    # Create constituency-level predictions
    train_constituency_results = train_results.groupby('Constit_Code').agg({
        'Actual_Vote': lambda x: pd.Series.mode(x)[0],
        'Predicted_Vote': lambda x: pd.Series.mode(x)[0]
    }).reset_index()
    
    test_constituency_results = test_results.groupby('Constit_Code').agg({
        'Actual_Vote': lambda x: pd.Series.mode(x)[0],
        'Predicted_Vote': lambda x: pd.Series.mode(x)[0]
    }).reset_index()
    
    #  Calculate detailed class performance

    class_performance = []

    for party in sorted(y_train.unique()):
    # Create binary labels for this party
        y_true_binary = (y_train == party)
        y_pred_binary = (train_pred == party)
    
        mask = y_train == party
        party_acc = accuracy_score(
            y_train[mask],
            train_pred[mask],
        )
        perf = {
            'Dataset': 'Train',
            'Party': party_names[party],
            'Count': mask.sum(),
            'Percentage': mask.sum()/len(y_train)*100,
            'Accuracy': party_acc,
            'Precision': precision_score(y_true_binary, y_pred_binary, zero_division= 0),
            'Recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'F1_Score': f1_score(y_true_binary, y_pred_binary, zero_division=0 )
        }
        class_performance.append(perf)

# Test performance
    for party in sorted(y_test.unique()):
    # Create binary labels for this party
        y_true_binary = (y_test == party)
        y_pred_binary = (test_pred == party)
    
        mask = y_test == party
        party_acc = accuracy_score(
            y_test[mask],
            test_pred[mask]
        )
        perf = {
            'Dataset': 'Test',
            'Party': party_names[party],
            'Count': mask.sum(),
            'Percentage': mask.sum()/len(y_test)*100,
            'Accuracy': party_acc,
            'Precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'Recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'F1_Score': f1_score(y_true_binary, y_pred_binary, zero_division=0)
        }
        class_performance.append(perf)

    class_performance_df = pd.DataFrame(class_performance)
    
    # Calculate confusion matrices
    train_cm = confusion_matrix(y_train, train_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    print("\nCalculating feature importance...")
    
    # Get coefficients for each class
    coefficients = model.coef_
    
    # Calculate mean absolute importance across all classes
    mean_importance = np.abs(coefficients).mean(axis=0)
    
    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean_Absolute_Importance': mean_importance,
        'Std_Importance': np.std(coefficients, axis=0)
    })
    
    # Add coefficients for each party
    for idx, party_id in enumerate(model.classes_):
        importance_df[f'Coefficient_{party_names[party_id]}'] = coefficients[idx]
    
    # Sort by mean absolute importance
    importance_df = importance_df.sort_values('Mean_Absolute_Importance', ascending=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(15), 
                x='Mean_Absolute_Importance', 
                y='Feature')
    plt.title('Top 15 Features by Mean Absolute Importance')
    plt.tight_layout()
    plt.savefig('lr_feature_importance.png')
    plt.close()

    
    # Save all results to Excel
    with pd.ExcelWriter('LR_model_results.xlsx') as writer:
        # Overall metrics
        pd.DataFrame({
            'Metric': ['Training Accuracy', 'Test Accuracy'],
            'Value': [train_accuracy, test_accuracy]
        }).to_excel(writer, sheet_name='Overall_Metrics', index=False)
        
        # Individual predictions
        train_results.to_excel(writer, sheet_name='Training_Predictions', index=False)
        test_results.to_excel(writer, sheet_name='Test_Predictions', index=False)
        
        # Constituency-level predictions
        train_constituency_results.to_excel(writer, sheet_name='Train_Constituency_Predictions', index=False)
        test_constituency_results.to_excel(writer, sheet_name='Test_Constituency_Predictions', index=False)
        
        # Class performance
        class_performance_df.to_excel(writer, sheet_name='Class_Performance', index=False)
        
        # Confusion matrices with party names
        pd.DataFrame(
            train_cm,
            columns=[party_names[i] for i in sorted(y_train.unique())],
            index=[party_names[i] for i in sorted(y_train.unique())]
        ).to_excel(writer, sheet_name='Train_Confusion_Matrix')
        
        pd.DataFrame(
            test_cm,
            columns=[party_names[i] for i in sorted(y_test.unique())],
            index=[party_names[i] for i in sorted(y_test.unique())]
        ).to_excel(writer, sheet_name='Test_Confusion_Matrix')
    
    print("\nResults saved to 'LR_model_results.xlsx'")
     # Additional CV results to save
    cv_results = pd.DataFrame(random_search.cv_results_)
    
    # Add CV results to Excel output
    with pd.ExcelWriter('LR_model_cv.xlsx') as writer:
        # Your existing Excel writing code...
        
        # Add CV results
        cv_results.to_excel(writer, sheet_name='CV_Results', index=False)

    
    
    return model, train_results, test_results, class_performance_df




def fit_multiclass_lr_balanced(train_df, test_df):
    """
    Fits a multi-class logistic regression model with constituency context
    for voting behavior prediction with balanced classes.
    """
    # Party mapping
    party_names = {
    1: 'Labour',    # 
    2: 'Conservatives',    
    3: 'Liberal Democrats',   
    4: 'Scottish National Party',    
    5: 'Other',    
    6: 'Didnt vote'
            }
    
    print("Fitting hierarchical model...")
    
    # Select predictors
    individual_predictors = [
        # Numeric/Ordinal variables
        'y01_Income', 'y09_Gender', 'y10_Age', 'y13_Education',
        
        # Employment dummies
        'employment_Homemaker', 'employment_Other', 'employment_Part-time',
        'employment_Retired', 'employment_Sick_leave', 'employment_Student',
        'employment_Unemployed',
        
        # Homeownership dummies
        'homeownership_Other', 'homeownership_Own_home_on_mortgage',
        'homeownership_Own_home_outright', 'homeownership_Rented',
        
        # Ethnicity dummies
        'ethnicity_Mixed', 'ethnicity_Asian', 'ethnicity_Black',
        'ethnicity_Arab', 'ethnicity_Other',
        
        # Religion dummies
        'religion_No religion', 'religion_Muslim', 'religion_Hindu',
        'religion_Sikh', 'religion_Buddhist', 'religion_Other'
    ]
    
    
    # Create clean copies of variables
    X_train = train_df[individual_predictors].copy()
    y_train = train_df['b02_Voting_Outcome'].copy()
    train_constituencies = train_df['Constit_Code'].copy()
    
    
    X_test = test_df[individual_predictors].copy()
    y_test = test_df['b02_Voting_Outcome'].copy()
    test_constituencies = test_df['Constit_Code'].copy()
  
    
    # Print initial class distribution
    print("\nClass distribution in training set:")
    train_dist = pd.Series(y_train).value_counts().sort_index()
    for party_id, count in train_dist.items():
        print(f"{party_names[party_id]}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Scale numeric variables
    numeric_vars = ['y13_Education', 'y01_Income', 'y10_Age']
    scaler = StandardScaler()
    X_train[numeric_vars] = scaler.fit_transform(X_train[numeric_vars])
    X_test[numeric_vars] = scaler.transform(X_test[numeric_vars])
    
    # Add constituency-level features
    train_constituency_stats = []
    for var in numeric_vars: 
        agg_stats = train_df.groupby('Constit_Code').apply(
            lambda x: np.average(x[var])
        ).reset_index()
        agg_stats.columns = ['Constit_Code', f'constituency_mean_{var}']
        train_constituency_stats.append(agg_stats)
    
    train_constituency_features = train_constituency_stats[0]
    for stats in train_constituency_stats[1:]:
        train_constituency_features = train_constituency_features.merge(
            stats, on='Constit_Code', how='outer'
        )
    
    # Same for test data
    test_constituency_stats = []
    for var in numeric_vars:
        agg_stats = test_df.groupby('Constit_Code').apply(
            lambda x: np.average(x[var])
        ).reset_index()
        agg_stats.columns = ['Constit_Code', f'constituency_mean_{var}']
        test_constituency_stats.append(agg_stats)
    
    test_constituency_features = test_constituency_stats[0]
    for stats in test_constituency_stats[1:]:
        test_constituency_features = test_constituency_features.merge(
            stats, on='Constit_Code', how='outer'
        )
    
    # Merge features
    X_train['Constit_Code'] = train_constituencies
    X_train = X_train.merge(train_constituency_features, on='Constit_Code', how='left')
    X_train = X_train.drop('Constit_Code', axis=1)
    
    X_test['Constit_Code'] = test_constituencies
    X_test = X_test.merge(test_constituency_features, on='Constit_Code', how='left')
    X_test = X_test.drop('Constit_Code', axis=1)

    # Define the parameter space
    param_distributions = {
        'C': loguniform(1e-3, 1e3),  
        'max_iter': [1000, 2000]
    }

    # Create base model
    base_model = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced')

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,  
        cv=5,  # 5-fold cross-validation
        scoring='balanced_accuracy',
        n_jobs=-1,  
        random_state=42,
        verbose=2
    )

    # Fit the random search
    print("Performing randomized search for optimal parameters...")
    random_search.fit(X_train, y_train)
    
    # Print results
    print("\nBest parameters found:")
    print(random_search.best_params_)
    print(f"\nBest cross-validation accuracy: {random_search.best_score_:.3f}")

    # Use the best model for predictions
    model = random_search.best_estimator_
    
    #Generate predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)
    
    #Calculate accuracies
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nOverall Accuracies:")
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")

    #8 Create dataframe
    train_results = pd.DataFrame({
        'Constit_Code': train_constituencies,
        'Actual_Vote': y_train,
        'Predicted_Vote': train_pred,
        'Age': train_df['y10_Age'],
        'Gender': train_df['y09_Gender'],
        'Education': train_df['y13_Education'],
        'Income': train_df['y01_Income'],
        'Region': train_df['region'] if 'region' in train_df.columns else None

    })
    
    test_results = pd.DataFrame({
        'Constit_Code': test_constituencies,
        'Actual_Vote': y_test,
        'Predicted_Vote': test_pred,
        'Age': test_df['y10_Age'],
        'Gender': test_df['y09_Gender'],
        'Education': test_df['y13_Education'],
        'Income': test_df['y01_Income'],
        'Region': test_df['region'] if 'region' in test_df.columns else None
        
    })
    
    # Add probability columns with party names
    for i, party in enumerate(model.classes_):
        train_results[f'Prob_{party_names[party]}'] = train_probs[:, i]
        test_results[f'Prob_{party_names[party]}'] = test_probs[:, i]
    
    # Create constituency-level predictions
    train_constituency_results = train_results.groupby('Constit_Code').agg({
        'Actual_Vote': lambda x: pd.Series.mode(x)[0],
        'Predicted_Vote': lambda x: pd.Series.mode(x)[0]
    }).reset_index()
    
    test_constituency_results = test_results.groupby('Constit_Code').agg({
        'Actual_Vote': lambda x: pd.Series.mode(x)[0],
        'Predicted_Vote': lambda x: pd.Series.mode(x)[0]
    }).reset_index()
    
    # Calculate detailed class performance
    class_performance = []
    
    for party in sorted(y_train.unique()):
    # Create binary labels for this party
        y_true_binary = (y_train == party)
        y_pred_binary = (train_pred == party)
    
        mask = y_train == party
        party_acc = accuracy_score(
            y_train[mask],
            train_pred[mask],
        )
        perf = {
            'Dataset': 'Train',
            'Party': party_names[party],
            'Count': mask.sum(),
            'Percentage': mask.sum()/len(y_train)*100,
            'Accuracy': party_acc,
            'Precision': precision_score(y_true_binary, y_pred_binary),
            'Recall': recall_score(y_true_binary, y_pred_binary),
            'F1_Score': f1_score(y_true_binary, y_pred_binary)
        }
        class_performance.append(perf)

# Test performance
    for party in sorted(y_test.unique()):
    # Create binary labels for this party
        y_true_binary = (y_test == party)
        y_pred_binary = (test_pred == party)
    
        mask = y_test == party
        party_acc = accuracy_score(
            y_test[mask],
            test_pred[mask]
        )
        perf = {
            'Dataset': 'Test',
            'Party': party_names[party],
            'Count': mask.sum(),
            'Percentage': mask.sum()/len(y_test)*100,
            'Accuracy': party_acc,
            'Precision': precision_score(y_true_binary, y_pred_binary),
            'Recall': recall_score(y_true_binary, y_pred_binary),
            'F1_Score': f1_score(y_true_binary, y_pred_binary)
        }
        class_performance.append(perf)

    class_performance_df = pd.DataFrame(class_performance)

    print("Unique values in y_train:", sorted(y_train.unique()))
    print("Unique values in y_test:", sorted(y_test.unique()))
    print("Party names keys:", sorted(party_names.keys()))  
    
    # Calculate confusion matrices
    train_cm = confusion_matrix(y_train, train_pred)
    test_cm = confusion_matrix(y_test, test_pred)

    print("\nCalculating feature importance...")
    
    # Get coefficients for each class
    coefficients = model.coef_
    
    # Calculate mean absolute importance across all classes
    mean_importance = np.abs(coefficients).mean(axis=0)
    
    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean_Absolute_Importance': mean_importance,
        'Std_Importance': np.std(coefficients, axis=0)
    })
    
    # Add coefficients for each party
    for idx, party_id in enumerate(model.classes_):
        importance_df[f'Coefficient_{party_names[party_id]}'] = coefficients[idx]
    
    # Sort by mean absolute importance
    importance_df = importance_df.sort_values('Mean_Absolute_Importance', ascending=False)
    
    # Create visualisations
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(15), 
                x='Mean_Absolute_Importance', 
                y='Feature')
    plt.title('Top 15 Features by Mean Absolute Importance')
    plt.tight_layout()
    plt.savefig('lr_feature_importance.png')
    plt.close()

    # Create heatmap of coefficients
    plt.figure(figsize=(12, 10))
    coef_columns = [col for col in importance_df.columns if col.startswith('Coefficient_')]
    heatmap_data = importance_df[['Feature'] + coef_columns].head(20).set_index('Feature')
    heatmap_data.columns = [col.replace('Coefficient_', '') for col in heatmap_data.columns]
    
    sns.heatmap(heatmap_data, 
                cmap='RdBu', 
                center=0,
                annot=True,
                fmt='.2f')
    plt.title('Top 20 Features: Coefficients by Party')
    plt.tight_layout()
    plt.savefig('lr_coefficient_heatmap.png')
    plt.close()



    
    # Save all results to Excel

    macro_precision = precision_score(y_test, test_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, test_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)

    weighted_precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)



    with pd.ExcelWriter('balanced_LR_model_results.xlsx') as writer:
        # Overall metrics
        pd.DataFrame({
        'Metric': ['Training Accuracy', 
                  'Test Accuracy',
                  'Macro Precision', 
                  'Macro Recall', 
                  'Macro F1',
                  'Weighted Precision', 
                  'Weighted Recall', 
                  'Weighted F1'],
        'Value': [train_accuracy, 
                 test_accuracy,
                 macro_precision, 
                 macro_recall, 
                 macro_f1,
                 weighted_precision, 
                 weighted_recall, 
                 weighted_f1]
    }).to_excel(writer, sheet_name='Overall_Metrics', index=False)
        # Individual predictions
        train_results.to_excel(writer, sheet_name='Training_Predictions', index=False)
        test_results.to_excel(writer, sheet_name='Test_Predictions', index=False)
        
        # Constituency-level predictions
        train_constituency_results.to_excel(writer, sheet_name='Train_Constituency_Predictions', index=False)
        test_constituency_results.to_excel(writer, sheet_name='Test_Constituency_Predictions', index=False)
        
        # Class performance
        class_performance_df.to_excel(writer, sheet_name='Class_Performance', index=False)
        
        # Confusion matrices with party names
        pd.DataFrame(
            train_cm,
            columns=[party_names[i] for i in sorted(y_train.unique())],
            index=[party_names[i] for i in sorted(y_train.unique())]
        ).to_excel(writer, sheet_name='Train_Confusion_Matrix')
        
        pd.DataFrame(
            test_cm,
            columns=[party_names[i] for i in sorted(y_test.unique())],
            index=[party_names[i] for i in sorted(y_test.unique())]
        ).to_excel(writer, sheet_name='Test_Confusion_Matrix')
        
        # Add feature importance
        importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        # Add top features by party
        top_features_by_party = pd.DataFrame()
        for party_id in sorted(y_train.unique()):
            coef_col = f'Coefficient_{party_names[party_id]}'
            
            # Get top 10 positive and negative features
            top_pos = importance_df.nlargest(10, coef_col)[['Feature', coef_col]]
            top_neg = importance_df.nsmallest(10, coef_col)[['Feature', coef_col]]
            
            top_pos['Party'] = party_names[party_id]
            top_pos['Direction'] = 'Positive'
            top_neg['Party'] = party_names[party_id]
            top_neg['Direction'] = 'Negative'
            
            top_features_by_party = pd.concat([top_features_by_party, top_pos, top_neg])
        
        top_features_by_party.to_excel(writer, sheet_name='Top_Features_By_Party', index=False)
    
    print("\nResults saved to 'balanced_LR_model_results.xlsx'")
     # Additional CV results to save
    cv_results = pd.DataFrame(random_search.cv_results_)
    
    # Add CV results to Excel output
    with pd.ExcelWriter('LR_model_balanced_cv.xlsx') as writer:
        
        # Add CV results
        cv_results.to_excel(writer, sheet_name='CV_Results', index=False)

    
    
    return model, train_results, test_results, class_performance_df












































