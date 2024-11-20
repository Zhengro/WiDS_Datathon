"""
A collection of utility functions for evaluating binary classification models, 
including feature importance, metrics on the validation set, and predictions for edge cases.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils.df_helpers import validate_column_exists


def validate_list_lengths(
        *lists) -> None:
    """
    Validate that all provided lists have the same length.

    Parameters:
    - *lists: Variable number of lists to validate.

    Raises:
    - ValueError: If the lengths of the lists are not the same.
    """
    if not lists:
        raise ValueError("At least one list must be provided.")

    for lst in lists[1:]:
        if len(lst) != len(lists[0]):
            raise ValueError(
                "The lengths of the provided lists must be the same.")


def rank_features_by_importance(
        feature_names: list[str],
        feature_importances: list[float],
        print_ranked_features: bool = False,
        num_features_to_print: int = 10) -> pd.DataFrame:
    """
    Ranks features by their importance and optionally prints the top and bottom features.

    Parameters:
    - feature_names (list[str]): A list of feature names.
    - feature_importances (list[float]): A list of corresponding feature importances.
    - print_ranked_features (bool): Whether to print the ranked features. Defaults to False.
    - num_features_to_print (int): Number of top and bottom features to print. Defaults to 10.

    Returns:
    - pd.DataFrame: A DataFrame containing the features ranked by importance.
    """
    validate_list_lengths(feature_names, feature_importances)

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    feature_importance.sort_values(
        by='Importance', ascending=False, inplace=True)

    if print_ranked_features:
        top_features = feature_importance.head(num_features_to_print)[
            'Feature'].tolist()
        bottom_features = feature_importance.tail(
            num_features_to_print)['Feature'].tolist()
        print(f'{num_features_to_print} Most Important Features: \n{top_features}')
        print(f'{num_features_to_print} Least Important Features: \n{
              bottom_features}')

    return feature_importance


def calculate_binary_classification_metrics(
        y_gt: List[int],
        y_pred: List[int],
        y_prob: List[float]) -> pd.DataFrame:
    """
    Calculate various metrics for evaluating a binary classification model.

    Parameters:
    - y_gt (List[int]): Ground truth labels (0 or 1).
    - y_pred (List[int]): Predicted labels (0 or 1).
    - y_prob (List[float]): Probability estimates for the positive class (1).

    Returns:
    - pd.DataFrame: DataFrame containing the evaluation metrics: 
    accuracy, precision, recall, f1_score, balanced_accuracy, and roc_auc.

    Metrics:
    - accuracy: 
        (TP + TN) / (TP + TN + FP + FN)
    - precision: 
        TP / (TP + FP)
        The ability of the classifier not to label as positive a sample that is negative.
    - recall/TPR: 
        TP / (TP + FN)
        The ability of the classifier to find all the positive samples.
    - f1_score: 
        2 * precision * recall / (precision + recall)
    - balanced_accuracy: 
        The average of recall obtained on each class.
    - roc_auc: 
        Calculated using TPR and FPR at various threshold settings.
        https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    """
    validate_list_lengths(y_gt, y_pred, y_prob)

    accuracy = accuracy_score(y_gt, y_pred)
    precision = precision_score(y_gt, y_pred, pos_label=1, average='binary')
    recall = recall_score(y_gt, y_pred, pos_label=1, average='binary')
    f1 = f1_score(y_gt, y_pred, pos_label=1, average='binary')
    balanced_accuracy = balanced_accuracy_score(y_gt, y_pred)
    roc_auc = roc_auc_score(y_gt, y_prob)

    metrics_df = pd.DataFrame(
        [[accuracy, precision, recall, f1, balanced_accuracy, roc_auc]],
        columns=['accuracy', 'precision', 'recall',
                 'f1_score', 'balanced_accuracy', 'roc_auc']
    )

    return metrics_df


def evaluate_binary_classifier(
        model: Union[LogisticRegression,
                     DecisionTreeClassifier,
                     RandomForestClassifier,
                     XGBClassifier],
        x_val: Union[pd.DataFrame, np.ndarray],
        y_val: pd.Series,
        print_report: bool = False,
        predict_edge_cases: bool = False,
        df_edge_cases: Optional[pd.DataFrame] = None,
        edge_case_prediction_indices:
        Optional[List[int]] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Evaluates a binary classifier model on validation data and 
    optionally prints a classification report and predicts edge cases.

    Parameters:
    - model: The trained binary classifier model. 
        Must be one of `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, 
        or `XGBClassifier`.
    - x_val (pd.DataFrame or np.ndarray): The validation features.
    - y_val (pd.Series): The ground-truth labels for the validation set.
    - print_report (bool): Whether to print the classification report. Defaults to False.
    - predict_edge_cases (bool): Whether to predict edge cases. Defaults to False.
    - df_edge_cases (Optional[pd.DataFrame]): 
        DataFrame containing edge cases. Required if `predict_edge_cases` is True.
    - edge_case_prediction_indices (Optional[List[int]]): 
        Indices of the edge cases in the validation set. Required if `predict_edge_cases` is True.

    Returns:
    - Tuple[pd.DataFrame, Optional[pd.DataFrame]]: 
        A tuple containing the scores DataFrame and 
        the updated edge cases DataFrame (if applicable).
    """
    y_pred = model.predict(x_val)
    y_prob = model.predict_proba(x_val)[:, 1]
    scores_df = calculate_binary_classification_metrics(y_val, y_pred, y_prob)

    if print_report:
        print(classification_report(y_val, y_pred))

    if predict_edge_cases:
        if df_edge_cases is None or edge_case_prediction_indices is None:
            raise ValueError(
                "Both df_edge_cases and edge_case_prediction_indices\
                    must be provided when predict_edge_cases is True.")
        validate_column_exists(df_edge_cases, 'partition')
        validate_column_exists(df_edge_cases, 'prediction')

        df_edge_cases.loc[df_edge_cases['partition'] == 'validation',
                          'prediction'] = y_pred[edge_case_prediction_indices]

    return scores_df, df_edge_cases
