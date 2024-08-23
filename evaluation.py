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
    """ Evaluation Metrics for Regression, Cross-Validation techniques and residuals """

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

    def residuals(self) -> np.ndarray:
        """Calculate the residuals."""
        return self.y_true - self.y_pred

    def abs_residuals(self) -> np.ndarray:
        """Calculate the absolute residuals."""
        return np.abs(self.residuals())

    def residual_mean(self) -> float:
        """Calculate the mean of the absolute residuals."""
        return np.mean(self.abs_residuals())

    def residual_max(self) -> float:
        """Calculate the maximum of the absolute residuals."""
        return np.max(self.abs_residuals())

    def residual_median(self) -> float:
        """Calculate the median of the absolute residuals."""
        return np.median(self.abs_residuals())

    def residual_std_dev(self) -> float:
        """Calculate the standard deviation of the residuals."""
        return np.std(self.abs_residuals())

    def MAE(self) -> float:
        """Calculate the Mean Absolute Error (MAE)."""
        return mean_absolute_error(self.y_true, self.y_pred)

    def RMSE(self) -> float:
        """Calculate the Root Mean Squared Error (RMSE)."""
        return root_mean_squared_error(self.y_true, self.y_pred)

    def HME(self) -> float:
        """Calculate the Harmonic Mean of Errors (HME)."""
        mae = self.MAE()
        rmse = self.RMSE()
        return 2 * (mae * rmse) / (mae + rmse)

    @staticmethod
    def AME(hme_loocv: float, hme_logocv: float) -> float:
        """Calculate the Arithmetic Mean of Errors (AME)."""
        return (hme_loocv + hme_logocv) / 2

    @staticmethod
    def average_metrics(metrics: list) -> 'EvaluationMetrics':
        """Average the metrics from a list of EvaluationMetrics instances."""
        mae_values = [m.MAE() for m in metrics]
        rmse_values = [m.RMSE() for m in metrics]
        hme_values = [m.HME() for m in metrics]
        residual_max_values = [m.residual_max() for m in metrics]
        residual_mean_values = [m.residual_mean() for m in metrics]
        residual_median_values = [m.residual_median() for m in metrics]
        residual_std_dev_values = [m.residual_std_dev() for m in metrics]

        avg_mae = np.mean(mae_values)
        avg_rmse = np.mean(rmse_values)
        avg_hme = np.mean(hme_values)
        avg_residual_max = np.mean(residual_max_values)
        avg_residual_mean = np.mean(residual_mean_values)
        avg_residual_median = np.mean(residual_median_values)
        avg_residual_std_dev = np.mean(residual_std_dev_values)

        # Create an empty EvaluationMetrics object with averaged metrics
        avg_metrics = EvaluationMetrics(
            y_true=[],
            y_pred=[]
        )

        # Monkey-patch the averaged values into the new EvaluationMetrics instance
        avg_metrics.MAE = lambda: avg_mae
        avg_metrics.RMSE = lambda: avg_rmse
        avg_metrics.HME = lambda: avg_hme
        avg_metrics.residual_max = lambda: avg_residual_max
        avg_metrics.residual_mean = lambda: avg_residual_mean
        avg_metrics.residual_median = lambda: avg_residual_median
        avg_metrics.residual_std_dev = lambda: avg_residual_std_dev

        return avg_metrics

    def __str__(self) -> str:
        return (
            "Evaluation Metrics:\n"
            f"\tMAE: {self.MAE():.2f}\n"
            f"\tRMSE: {self.RMSE():.2f}\n"
            f"\tHME: {self.HME():.2f}\n"
            f"\tResidual Max: {self.residual_max():.2f}\n"
            f"\tResidual Mean: {self.residual_mean():.2f}\n"
            f"\tResidual Median: {self.residual_median():.2f}\n"
            f"\tResidual Std. Dev.: {self.residual_std_dev():.2f}\n"
        )
