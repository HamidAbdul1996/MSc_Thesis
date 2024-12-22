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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier 
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns


def create_train_test_split(df, test_size=0.2, random_state=42):
    """
    Create train-test split while keeping constituencies together
    
    Parameters:
    -----------
    df: DataFrame
        Preprocessed data
    test_size: float
        Proportion of data to use for testing (default 0.2)
    random_state: int
        Random seed for reproducibility
    
    Returns:
    --------
    train_df, test_df: DataFrames for training and testing
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



def fit_multiclass_gb_balanced(train_df, test_df):
    """
     Fits a Gradient Boosting model for multi-party voting prediction with constituency context.
    Includes cross-validation. Classes are balanced.
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
    
    print("Fitting gradient boosting model...")
    
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
  
     # At the beginning of your function, after getting y_train:
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
  
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

    # Create base model
    param_distributions = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.5, 0.7, 1.0] 
    }

    base_model = GradientBoostingClassifier(
    random_state=42)

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

    sample_weights = np.array([class_weight_dict[y] for y in y_train])    # Fit the random search
    print("Performing randomized search for optimal parameters...")
    random_search.fit(X_train, y_train, sample_weight = sample_weights)
    
    # Print results
    print("\nBest parameters found:")
    print(random_search.best_params_)
    print(f"\nBest cross-validation accuracy: {random_search.best_score_:.3f}")

    # Use the best model for predictions
    model = random_search.best_estimator_
    
    
    # Generate predictions
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
    
    #feautre importance 
    print("\nCalculating feature importance...")
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    # Built-in feature importance
    builtin_importance = model.feature_importances_
    
    # Permutation importance (more reliable but slower)
    perm_importance = permutation_importance(
        model, X_train, y_train,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Built_in_Importance': builtin_importance,
        'Permutation_Importance_Mean': perm_importance.importances_mean,
        'Permutation_Importance_Std': perm_importance.importances_std
    })
    
    # Sort by built-in importance
    importance_df = importance_df.sort_values('Built_in_Importance', ascending=False)
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Plot built-in importance
    sns.barplot(
        data=importance_df.head(15),
        x='Built_in_Importance',
        y='Feature',
        ax=ax1
    )
    ax1.set_title('Top 15 Features by Built-in Importance')
    
    # Plot permutation importance
    sns.barplot(
        data=importance_df.head(15),
        x='Permutation_Importance_Mean',
        y='Feature',
        ax=ax2
    )
    # Add error bars for permutation importance
    ax2.errorbar(
        x=importance_df.head(15)['Permutation_Importance_Mean'],
        y=range(15),
        xerr=importance_df.head(15)['Permutation_Importance_Std'],
        fmt='none',
        c='black',
        alpha=0.3
    )
    ax2.set_title('Top 15 Features by Permutation Importance')
    
    plt.tight_layout()
    plt.savefig('gb_feature_importance.png')
    plt.close()
    
   
    macro_precision = precision_score(y_test, test_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, test_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)
    weighted_precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)


    with pd.ExcelWriter('balanced_GB_model_results.xlsx') as writer:
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
    # 12. Save all results to Excel
        
        # Individual predictions
        train_results.to_excel(writer, sheet_name='Training_Predictions', index=False)
        test_results.to_excel(writer, sheet_name='Test_Predictions', index=False)
        
        # Constituency-level predictions
        train_constituency_results.to_excel(writer, sheet_name='Train_Constituency_Predictions', index=False)
        test_constituency_results.to_excel(writer, sheet_name='Test_Constituency_Predictions', index=False)
        
        # Class performance
        class_performance_df.to_excel(writer, sheet_name='Class_Performance', index=False)

        importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        # Add top features comparison
        top_features_comparison = pd.DataFrame({
            'Rank': range(1, len(feature_names) + 1),
            'Top_by_Built_in': importance_df.sort_values('Built_in_Importance', ascending=False)['Feature'],
            'Top_by_Permutation': importance_df.sort_values('Permutation_Importance_Mean', ascending=False)['Feature']
        })
        top_features_comparison.to_excel(writer, sheet_name='Feature_Importance_Comparison', index=False)
        
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
    
    print("\nResults saved to 'balanced_GB_model_results.xlsx'")
     # Additional CV results to save
    cv_results = pd.DataFrame(random_search.cv_results_)
    
    # Add CV results to Excel output
    with pd.ExcelWriter('balanced_GB_model_cv.xlsx') as writer:
        
        
        # Add CV results
        cv_results.to_excel(writer, sheet_name='CV_Results', index=False)

    
    
    return model, train_results, test_results, class_performance_df