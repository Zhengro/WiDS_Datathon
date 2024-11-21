from typing import Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CustomLabelEncoder(LabelEncoder):
    """
    A custom label encoder that extends sklearn's LabelEncoder to handle unseen labels.

    Attributes:
        unseen_label (str): The label to use for unseen categories.
        unseen_code (int | None): The encoded value for the unseen label.
        known_labels (Set[str] | None): A set of all known labels including the unseen label.
    """

    def __init__(self) -> None:
        """
        Initializes the CustomLabelEncoder with an unseen label.
        """
        super().__init__()
        self.unseen_label: str = 'unseen'
        self.unseen_code: int | None = None
        self.known_labels: Set[str] | None = None

    def fit(
            self,
            y: pd.Series) -> 'CustomLabelEncoder':
        """
        Fits the encoder to the provided labels and includes the unseen label in the encoding.

        Parameters:
        - y (pd.Series): The labels to fit.

        Returns:
        - CustomLabelEncoder: The fitted encoder instance.
        """
        super().fit(list(y) + [self.unseen_label])
        self.unseen_code = super().transform([self.unseen_label])[0]
        self.known_labels = set(list(y) + [self.unseen_label])

        return self

    def transform(
            self,
            y: pd.Series) -> pd.Series:
        """
        Transforms the provided labels into their corresponding encoded values.
        Unseen labels are mapped to the unseen code.

        Parameters:
        - y (pd.Series): The labels to transform.

        Returns:
        - np.ndarray: An array of encoded labels.

        Raises:
        - ValueError: If the encoder has not been fitted yet.
        """
        if self.unseen_code is None:
            raise ValueError(
                "The encoder has not been fitted yet. Please call fit before transform.")

        y = np.array(y)
        y = np.where(np.isin(y, list(self.known_labels)), y, self.unseen_label)
        transformed = super().transform(y)

        return transformed
