"""
Evaluation Module
Evaluation Metrics for Regression and Cross-Validation Techniques

This module implements various evaluation metrics to assess the performance of predictive models, particularly in regression and cross-validation contexts.

Author: Mariano Garralda-Barrio, Carlos Eiras-Franco, Verónica Bolón-Canedo
License: MIT License (see LICENSE file for details)
Date: 2024

Usage:
- Import the `EvaluationMetrics` class and use the `MAE`, `RMSE`, `HME`, or `AME` methods to calculate evaluation metrics.
- See the documentation or README for more details on how to use this module.

Example:
    y_true = [values...]
    y_pred = [values...]

    metrics = EvaluationMetrics(y_true, y_pred)
    print(f"Metrics: {metrics}")
    
    print(f"MAE: {metrics.MAE():.2f}")
    print(f"RMSE: {metrics.RMSE():.2f}")
    print(f"HME: {metrics.HME():.2f}")

    hme_loocv = metrics.HME()
    hme_logocv = metrics.HME()

    ame = EvaluationMetrics.AME(hme_loocv, hme_logocv)
    print(f"AME: {ame:.2f}")
"""

import array
import warnings
from typing import List
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)


class EvaluationMetrics:
    """ Evaluation Metrics for Regression and Cross-Validation techniques """

    def __init__(self, y_true: array, y_pred: array) -> None:
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)

        # Create a mask to filter out possible NaN values from y_pred
        nan_mask = np.isnan(self.y_pred)
        nan_count = np.sum(nan_mask)

        if nan_count > 0:
            warnings.warn(f"Warning: {nan_count} NaN values found in y_pred. These will be ignored in the evaluation metric calculations.",
                          UserWarning)

            # Apply the mask to both y_true and y_pred
            self.y_true = self.y_true[~nan_mask]
            self.y_pred = self.y_pred[~nan_mask]

    def MAE(self) -> float:
        """ Mean Absolute Error """
        return mean_absolute_error(self.y_true, self.y_pred)

    def RMSE(self) -> float:
        """ Root Mean Squared Error """
        return root_mean_squared_error(self.y_true, self.y_pred)

    def HME(self) -> float:
        """ Harmonic Mean Errors """
        return 2 * (self.MAE() * self.RMSE()) / (self.MAE() + self.RMSE())

    @staticmethod
    def AME(hme_loocv: float, hme_logocv: float) -> float:
        """ Arithmetic Mean Errors """
        return (hme_loocv + hme_logocv) / 2

    @staticmethod
    def average_metrics(metrics: List['EvaluationMetrics']) -> 'EvaluationMetrics':
        """ Average the metrics of a list of EvaluationMetrics """

        # Calculate the average of each metric individually
        mae_values = [m.MAE() for m in metrics]
        rmse_values = [m.RMSE() for m in metrics]
        hme_values = [m.HME() for m in metrics]

        avg_mae = np.average(mae_values)
        avg_rmse = np.average(rmse_values)
        avg_hme = np.average(hme_values)

        # Create an empty EvaluationMetrics object with averaged metrics
        avg_metrics = EvaluationMetrics(
            y_true=[],
            y_pred=[]
        )

        # Monkey-patch the averaged values into the new EvaluationMetrics instance
        avg_metrics.MAE = lambda: avg_mae
        avg_metrics.RMSE = lambda: avg_rmse
        avg_metrics.HME = lambda: avg_hme

        return avg_metrics

    def __str__(self) -> str:
        return (
            "Evaluation Metrics:\n"
            f"\tMAE: {self.MAE():.2f}\n"
            f"\tRMSE: {self.RMSE():.2f}\n"
            f"\tHME: {self.HME():.2f}\n"
        )
