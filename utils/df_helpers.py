"""
A collection of utility functions for performing operations on Pandas DataFrames, 
including validation, statistical analysis, and visualization.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def validate_column_exists(
        df: pd.DataFrame,
        col: str) -> None:
    """
    Validates if the specified column exists in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - col (str): The name of the column to check.

    Raises:
    - ValueError: If the specified column does not exist in the DataFrame.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' does not exist in the DataFrame.")


def find_duplicates(
        df: pd.DataFrame,
        id_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Identifies and filters duplicates based on all columns except the specified ID column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - id_col (str): The name of the ID column.

    Returns:
    - Tuple[pd.DataFrame, List[str]]: A tuple containing:
        - pd.DataFrame: A DataFrame containing only the duplicated rows.
        - List[str]: List of columns to check for duplicates.
    """
    if df.empty:
        return pd.DataFrame(), []
    validate_column_exists(df, id_col)

    columns_to_check = df.columns.difference([id_col]).to_list()
    duplicates_mask = df.duplicated(subset=columns_to_check, keep=False)
    duplicates_df = df[duplicates_mask]

    return duplicates_df, columns_to_check


def get_value_counts(
        df: pd.DataFrame,
        col: str,
        normalize: bool = True,
        dropna: bool = False) -> pd.DataFrame:
    """
    Computes the value counts for a specified column in a DataFrame, 
    optionally normalizing and dropping missing values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - col (str): The name of the column to compute value counts for.
    - normalize (bool, optional): 
        If True, the value counts will be normalized to percentages. Defaults to True.
    - dropna (bool, optional): 
        If True, missing values will be dropped from the value counts. Defaults to False.

    Returns:
    - pd.DataFrame: A DataFrame containing the value counts (multiplied by 100 if normalized).
    """
    if df.empty:
        return pd.DataFrame()
    validate_column_exists(df, col)

    value_counts = df[col].value_counts(normalize=normalize, dropna=dropna)
    if normalize:
        value_counts *= 100

    return value_counts.to_frame()


def get_combined_value_counts(
        df: pd.DataFrame,
        col: str,
        filter_col: str,
        filter_values: List,
        normalize: bool = True,
        dropna: bool = False) -> pd.DataFrame:
    """
    Computes and combines the value counts for a specified column 
    in the original DataFrame and filtered DataFrames.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - col (str): The name of the column to compute value counts for.
    - filter_col (str): The name of the column to filter on.
    - filter_values (List): A list of values to filter by in the filter_col.
    - normalize (bool, optional): 
        If True, the value counts will be normalized to percentages. Defaults to True.
    - dropna (bool, optional): 
        If True, missing values will be dropped from the value counts. Defaults to False.

    Returns:
    - pd.DataFrame: A DataFrame containing the combined value counts.
    """
    if df.empty:
        return pd.DataFrame()
    validate_column_exists(df, col)
    validate_column_exists(df, filter_col)

    original_counts = get_value_counts(
        df, col, normalize=normalize, dropna=dropna)
    original_counts.columns = ['All (%)']
    combined_df = original_counts.copy()

    for value in filter_values:
        filtered_df = df[df[filter_col] == value]
        filtered_counts = get_value_counts(
            filtered_df, col, normalize=normalize, dropna=dropna)
        filtered_counts.columns = [f'{filter_col}={value} (%)']
        combined_df = combined_df.join(filtered_counts, how='left')

    return combined_df


def highlight_diff(
        df: pd.DataFrame,
        col1: str,
        col2: str) -> pd.DataFrame:
    """
    Highlight rows where the values in the specified columns are different for non-missing rows.

    Parameters:
    - df (pd.DataFrame): The DataFrame to apply the styling to.
    - col1 (str): The name of the first column to compare.
    - col2 (str): The name of the second column to compare.

    Returns:
    - pd.DataFrame: A DataFrame with applied styles.
    """
    def highlight_row(row):
        if (row[col1] != row[col2]) and (pd.notna(row[col1])) and (pd.notna(row[col2])):
            return ['background-color: yellow'] * len(row)
        return [''] * len(row)

    if df.empty:
        return df
    validate_column_exists(df, col1)
    validate_column_exists(df, col2)

    return df.style.apply(highlight_row, axis=1)


def highlight_nan(
        df: pd.DataFrame) -> pd.DataFrame:
    """
    Highlight NaN values in the DataFrame and rows where the index is NaN.

    Parameters:
    - df (pd.DataFrame): The DataFrame to apply the styling to.

    Returns:
    - pd.DataFrame: A DataFrame with applied styles.
    """
    def highlight_row(row):
        if pd.isna(row.name):
            return ['background-color: gray'] * len(row)
        return ['background-color: gray' if pd.isna(v) else '' for v in row]

    if df.empty:
        return df

    return df.style.apply(highlight_row, axis=1)


def plot_category_distribution(
        df: pd.DataFrame,
        cat_col: str,
        index_labels: Optional[List[Tuple[int, str]]] = None,
        title_fontsize: int = 14,
        tick_fontsize: int = 10) -> None:
    """
    Plots the distribution of a categorical column in a DataFrame, 
    normalizing the value counts and optionally setting a specific order.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - cat_col (str): The name of the categorical column to plot.
    - index_labels (Optional[List[Tuple[int, str]]]): 
        A list of tuples where each inner tuple contains an index and its corresponding label.
    - title_fontsize (int, optional): Font size for the plot title. Defaults to 14.
    - tick_fontsize (int, optional): Font size for the plot ticks. Defaults to 10.
    """
    if df.empty:
        return
    validate_column_exists(df, cat_col)

    series = df[cat_col]
    series.name = None
    value_counts_normalized = series.value_counts(normalize=True, dropna=False)

    if index_labels:
        sorted_index_labels = sorted(index_labels)
        order = [index for index, _ in sorted_index_labels]
        missing_categories = set(value_counts_normalized.index) - set(order)
        if missing_categories:
            raise ValueError(
                f"The following categories are missing from index_labels: {missing_categories}")
        value_counts_normalized = value_counts_normalized.reindex(order)

    value_counts_normalized.plot(kind='barh', figsize=(
        10, min(len(value_counts_normalized.index), 8)))
    plt.title(f'{cat_col} Distribution (%)', fontsize=title_fontsize)
    plt.xlim([0, 1])
    if index_labels:
        plt.yticks(ticks=range(len(order)),
                   labels=[f'{index} ({label})' for index,
                           label in sorted_index_labels],
                   fontsize=tick_fontsize)
    plt.show()


def plot_histogram_with_percentages(
        df: pd.DataFrame,
        num_col: str,
        bins: Union[int, List[int]],
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        text_fontsize: int = 10,
        tick_fontsize: int = 10) -> None:
    """
    Plot a histogram for a specified numerical column of a DataFrame, 
    showing the percentage each bin represents.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - num_col (str): The name of the numerical column to plot.
    - bins (int or List[int]): The number of bins or the boundaries of the bins.
    - title_fontsize (int, optional): Font size for the plot title. Defaults to 14.
    - label_fontsize (int, optional): Font size for the plot labels. Defaults to 12.
    - text_fontsize (int, optional): Font size for the text annotations. Defaults to 10.
    - tick_fontsize (int, optional): Font size for the plot ticks. Defaults to 10.
    """
    if df.empty:
        return
    validate_column_exists(df, num_col)
    if not pd.api.types.is_numeric_dtype(df[num_col]):
        raise ValueError(f"Column '{num_col}' is not numerical.")
    if df[num_col].isna().any():
        raise ValueError(
            f"Column '{num_col}' contains missing values. Remove them before plotting.")

    total_count = len(df[num_col])
    weights = [100 / total_count] * total_count

    plt.figure(figsize=(10, 6))
    hist = sns.histplot(x=df[num_col], bins=bins, weights=weights, kde=False)
    for p in hist.patches:
        height = p.get_height()
        if height > 0:
            hist.text(p.get_x() + p.get_width() / 2., height + 0.5,
                      f'{height:.1f}%', ha="center", fontsize=text_fontsize)
    plt.title(f'{num_col} Distribution', fontsize=title_fontsize)
    plt.xlabel(num_col, fontsize=label_fontsize)
    plt.ylabel('Percentage', fontsize=label_fontsize)
    if isinstance(bins, int):
        bins = pd.cut(df[num_col], bins=bins, retbins=True)[1]
    plt.xticks(ticks=bins, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.show()


def plot_side_by_side_histogram_with_percentages(
        df: pd.DataFrame,
        num_col: str,
        filter_col: str,
        filter_values: List[str],
        bins: Union[int, List[int]],
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        text_fontsize: int = 10,
        tick_fontsize: int = 10) -> None:
    """
    Plot a side-by-side histogram for a specified numerical column of a DataFrame, 
    showing the percentage each bin represents for both the original and filtered distributions.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - num_col (str): The name of the numerical column to plot.
    - filter_col (str): The name of the column to filter on.
    - filter_values (List[str]): A list of values to filter by in the filter_col.
    - bins (int or List[int]): The number of bins or the boundaries of the bins.
    - title_fontsize (int, optional): Font size for the plot title. Defaults to 14.
    - label_fontsize (int, optional): Font size for the plot labels. Defaults to 12.
    - text_fontsize (int, optional): Font size for the text annotations. Defaults to 10.
    - tick_fontsize (int, optional): Font size for the plot ticks. Defaults to 10.
    """
    if df.empty:
        return
    validate_column_exists(df, num_col)
    validate_column_exists(df, filter_col)
    if not pd.api.types.is_numeric_dtype(df[num_col]):
        raise ValueError(f"Column '{num_col}' is not numerical.")
    if df[num_col].isna().any():
        raise ValueError(
            f"Column '{num_col}' contains missing values. Remove them before plotting.")

    if isinstance(bins, int):
        bins = pd.cut(df[num_col], bins=bins, retbins=True)[1]
    bins = np.array(bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = bins[1:] - bins[:-1]
    counts_df = pd.DataFrame()
    original_counts, _ = np.histogram(df[num_col], bins=bins)
    counts_df['All'] = original_counts
    for value in filter_values:
        filtered_df = df[df[filter_col] == value]
        filtered_counts, _ = np.histogram(filtered_df[num_col], bins=bins)
        counts_df[f'{filter_col}={value}'] = filtered_counts
    counts_df = counts_df.div(counts_df.sum(axis=0)) * 100

    _, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(counts_df.columns))
              for i in range(len(counts_df.columns))]
    for i, (col, color) in enumerate(zip(counts_df.columns, colors)):
        for j, (center, width, count) in enumerate(zip(bin_centers, bin_widths, counts_df[col])):
            x = center + (i - len(counts_df.columns) // 2) * \
                (width / len(counts_df.columns))
            ax.bar(x=x, height=count, width=width / (len(counts_df.columns)),
                   alpha=0.7, color=color, label=col if j == 0 else "")
            ax.text(x=x, y=count + 0.5,
                    s=f'{count:.1f}', ha="center", fontsize=text_fontsize)
    plt.title(f'{num_col} Distribution', fontsize=title_fontsize)
    plt.xlabel(num_col, fontsize=label_fontsize)
    plt.ylabel('Percentage', fontsize=label_fontsize)
    plt.xticks(ticks=bins, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend()
    plt.show()
