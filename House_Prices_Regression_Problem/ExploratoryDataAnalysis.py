import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import shap
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import f_oneway, chi2_contingency, pearsonr, spearmanr

# Correlataion between numeric features and numeric target, pearson spearman.
def correlation_numeric(df, target, method = 'pearson'):
    num_cols = df.drop(columns = target).select_dtypes(include = ['number']).columns
    corr_matrix = pd.DataFrame(index=num_cols, columns=['Correlation', 'P-Value'])
    
    for col in num_cols:
        if method == 'pearson':
            corr, p_value = pearsonr(df[col], df[target])
        elif method == 'spearman':
            corr, p_value = spearmanr(df[col], df[target])
        corr_matrix.loc[col] = [corr, p_value]
    
    return corr_matrix

# Relationship between numeric features and categoric target. ANOVA test for each categoric feature.
def anova(df, target):
    cat_cols = df.drop(columns = target).select_dtypes(include = ['object', 'category']).columns
    results = []
    for cat in cat_cols:
        groups = [group[target].values for name, group in df.groupby(cat)]
        f_scores, p_values = f_oneway(*groups)
        results.append((cat, f_scores, p_values))
    results_df = pd.DataFrame(results, columns=['Categorical Feature', 'F-Score', 'P-Value'])
    return results_df

# Relationships between categoric features snd categorical target. Chi square.
def chi_square(df, target):
    cat_cols = df.drop(columns = target).select_dtypes(include = ['object', 'category']).columns
    results = []
    for cat in cat_cols:
        contingency_table = pd.crosstab(df[cat], df[target])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        results.append((cat, chi2, p))
    results_df = pd.DataFrame(results, columns=['Categorical Feature', 'Chi-Square', 'P-Value'])
    return results_df

# Relationships between numeric features and categorical target.
def correlation_categoric(df, target, method = 'pointbiserial'):
    num_cols = df.drop(columns = target).select_dtypes(include = ['number']).columns
    corr_matrix = pd.DataFrame(index=num_cols, columns=['Correlation', 'P-Value'])
    
    for col in num_cols:
        if method == 'pointbiserial':
            corr, p_value = pearsonr(df[col], df[target].astype('category').cat.codes)
        elif method == 'spearman':
            corr, p_value = spearmanr(df[col], df[target].astype('category').cat.codes)
        corr_matrix.loc[col] = [corr, p_value]
    
    return corr_matrix

# SHAP tests for feature importance
def shap_feature_importance(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)
    return shap_values