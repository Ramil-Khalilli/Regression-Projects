import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats

# clean and readable column names
def column_name_cleaning(df):
    """
    Clean and standardize DataFrame column names.

    Steps:
    - Strip leading/trailing spaces
    - Add underscores between camelCase letters
    - Replace special characters with underscores
    - Convert all names to lowercase

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.Index: Cleaned column names
    """
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r'([a-z])([A-Z])', r'\1_\2', regex = True)
    df.columns = df.columns.str.replace('[ ?!@#$*]', '_', regex = True)
    df.columns = df.columns.str.lower()
    return df.columns


# clean and readable string values 


# show missing values as table and heatmap
def show_missing(df, table = True):
    """
    Display missing values in the DataFrame.

    Options:
    - Table: shows number and percentage of missing values per column
    - Heatmap: visual representation of missing values

    Args:
        df (pd.DataFrame): Input DataFrame
        table (bool): If True, return table; if False, show heatmap

    Returns:
        pd.DataFrame or sns.heatmap: Table or heatmap of missing values
    """
    if table:
        value = df.isnull().sum().values
        value_percent = ((df.isnull().sum())*100/len(df)).round().astype('int').values
        return pd.DataFrame({'Missing':value, 'Missing %':value_percent}, index=df.columns)
    elif table == False:
        return sns.heatmap(df.isnull())


# handle missing values 
def drop_missing(df, subset = None):
    """
    Drop rows with missing values in specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): List of columns to check for missing values. Default is None (all columns).

    Returns:
        str: Confirmation message
    """
    df = df.dropna(subset = subset)
    return 'Rows with missing values dropped'


# check duplicates
def show_duplicates(df):
    """
    Count duplicated rows in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        int: Number of duplicated rows
    """
    return df.duplicated().sum()

def remove_duplicates(df):
    """
    Remove duplicated rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        str: Confirmation message
    """
    df = df[~df.duplicated()]
    return "Duplicates Removed!"


# show outliers/extremes as table and box-plots
def show_outliers(df):
    """
    Identify outliers in numeric columns using the IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.Series: Number of outliers per numeric column
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    outlier_count = ((df < lower_bound) | (df > upper_bound)).sum(axis = 0)
    return outlier_count


# handle outliers: clip , drop
def handle_outliers(df, columns=None, method='drop'):
    """
    Handle outliers in numeric columns using either clipping or dropping.
    Works even if columns contain NaNs.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list or pd.Index, optional): Columns to check for outliers. Default is all numeric columns.
        method (str): 'drop' to remove outliers, 'clip' to cap them at bounds

    Returns:
        pd.DataFrame: DataFrame with outliers handled
    """
    import numpy as np

    if method not in ['drop', 'clip']:
        raise ValueError(f"'{method}' is not a valid method, please choose from ['drop', 'clip']")

    # Ensure columns is a list
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    elif isinstance(columns, (str, pd.Index, np.ndarray)):
        columns = list(columns) if not isinstance(columns, list) else columns

    df = df.copy()

    if method == 'drop':
        mask = pd.Series(False, index=df.index)
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        df = df[~mask].reset_index(drop=True)

    elif method == 'clip':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Clip each column individually
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df





# show distribution of numeric features 
def numeric_distribution(df, columns = None):
    """
    Plot histogram distributions of numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Columns to plot. Default is all numeric columns.

    Returns:
        None
    """
    if columns is None:
        columns = df.select_dtypes(exclude = ['object', 'category']).columns
    for col in columns:
        plt.figure(figsize = (12, 8))
        sns.histplot(data = df, x = col)


# scale numeric features, method=minmax, standard, robust
def scaling(df, columns = None, method = 'standard'):
    """
    Scale numeric features using Standard, MinMax, or Robust scaling.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Columns to scale. Default is all numeric columns.
        method (str): Scaling method: 'standard', 'minmax', or 'robust'

    Returns:
        pd.DataFrame: DataFrame with scaled columns
    """
    if columns is None:
        columns = df.select_dtypes(exclude = ['object', 'category']).columns
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }

    if method.lower() not in scalers:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(scalers.keys())}.")

    scaler = scalers[method.lower()]
    df[columns] = scaler.fit_transform(df[columns])

    return df


# encode categoric features, method=onehot, dummy, label, ordinalencoder
def categorical_encoding(df, columns = None, method = 'onehot', categories = None):
    """
    Encode categorical features using one-hot, dummy, label, or ordinal encoding.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to encode
        method (str): 'onehot', 'dummy', 'label', or 'ordinal'
        categories (list, optional): Categories for ordinal encoding

    Returns:
        pd.DataFrame: DataFrame with encoded columns
    """
    if method == 'onehot':
        df = pd.get_dummies(df, columns=columns, drop_first=False)
    elif method == 'dummy':
        df = pd.get_dummies(df, columns = columns, drop_first=True)
    elif method == 'label':
        encoder = LabelEncoder()
        for col in columns:
            df[col] = encoder.fit_transform(df[col])
    elif method == 'ordinal':
        encoder = OrdinalEncoder(categories=categories)
        df[columns] = encoder.fit_transform(df[columns])
    return df
