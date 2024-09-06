"""
Validation Module
Cross-Validation Techniques for Performance Evaluation of Big Data Workloads

This module implements various cross-validation techniques to evaluate the performance of predictive models for big data workloads, particularly in Apache Spark environments.

Author: Mariano Garralda-Barrio, Carlos Eiras-Franco, Verónica Bolón-Canedo
License: MIT License (see LICENSE file for details)
Date: 2024

Usage:
- Import the `CrossValidation` class and use the `leave_one_out` or `leave_one_group_out` methods to perform cross-validation.
- See the documentation or README for more details on how to use this module.

Example:
    validation = CrossValidation(
        w=garralda_workload_descriptors,
        w_groups=worload_groups,
        cs=configuration_settings
        t=execution_times,
        k_min=k_min,
        k_max=k_max
        eebc_weight=0.5
    )

    loocv_eval = validation.leave_one_out()
    logocv_eval = validation.leave_one_group_out()

    print(f"{loocv_eval}")
    print(f"{logocv_eval}")

    ame = EvaluationMetrics.AME(
        loocv_eval.HME(),
        logocv_eval.HME()
    )

    print(f"AME: {ame:.2f}")
"""

import warnings
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from performance.domain.model.garralda_perf_model import GarraldaPerformanceModel
from evaluation import EvaluationMetrics


class CrossValidation:
    """ Cross validation techniques """

    def __init__(
            self,
            w: np.ndarray,
            w_groups: np.ndarray,
            cs: np.ndarray,
            t: np.ndarray,
            k_min: int,
            k_max: int,
            eebc_weight: float
    ) -> None:
        """ Initialize the cross validation attributes
        Args:
            w (np.ndarray): Garralda workload descriptor.
            w_groups (np.ndarray): Unique groups of workloads.
            cs (np.ndarray): Configuration settings.
            t (np.ndarray): Execution times (target).
            k_min (int): Minimum value in range K.
            k_max (int): Maximum value in range K.
            eebc_weight (float): Weight to use on EEBC. Default is 0.5.
        """

        self.w = w
        self.w_groups = w_groups
        self.cs = cs
        self.t = t
        self.k_min = k_min
        self.k_max = k_max
        self.eebc_weight = eebc_weight

        self.perf_model = GarraldaPerformanceModel()

    def leave_one_out(self) -> EvaluationMetrics:
        """
        Leave-one-out cross-validation

        :return: EvaluationMetrics
        """

        y_pred = []
        y_true = []

        loocv = LeaveOneOut()
        # Split the data into training and test sets for each fold leaving one sample out
        for train_index, test_index in loocv.split(X=self.w):
            w_train, w_test = self.w[train_index], self.w[test_index]
            cs_train, cs_test = self.cs[train_index], self.cs[test_index]
            t_train, t_test = self.t[train_index], self.t[test_index]

            try:
                # Lazy training and prediction
                prediction = self.perf_model.fit_predict(
                    workload_descriptors=w_train,
                    configuration_settings=cs_train,
                    execution_times=t_train,
                    workload_ref=w_test,
                    new_configuration_settings=cs_test,
                    k_min=self.k_min,
                    k_max=self.k_max,
                    eebc_weight=self.eebc_weight
                )

                y_pred.append(prediction)
                y_true.append(t_test)

            except Exception as e:
                warnings.warn(f"leave_one_group_out: {e}", UserWarning)

        evaluation_metrics = EvaluationMetrics(y_true, y_pred)

        return evaluation_metrics

    def leave_one_group_out(self) -> EvaluationMetrics:
        """
        Leave-one-group-out cross-validation

        :return: EvaluationMetrics
        """

        evaluation_metrics = []

        logocv = LeaveOneGroupOut()
        # Split the data into training and test sets for each fold leaving one workload group out
        for train_index, test_index in logocv.split(X=self.w, groups=self.w_groups):
            y_pred = []
            y_true = []

            w_train, w_test = self.w[train_index], self.w[test_index]
            cs_train, cs_test = self.cs[train_index], self.cs[test_index]
            t_train, t_test = self.t[train_index], self.t[test_index]

            # For each sample (workload_ref) in test_data (workload group) a lazy model is trained to predict the execution time
            for i in range(len(w_test)):
                w_test_i = w_test[i]  # workload_ref
                cs_test_i = cs_test[i]
                t_test_i = t_test[i]

                try:
                    prediction = self.perf_model.fit_predict(
                        workload_descriptors=w_train,
                        configuration_settings=cs_train,
                        execution_times=t_train,
                        workload_ref=w_test_i,
                        new_configuration_settings=cs_test_i,
                        k_min=self.k_min,
                        k_max=self.k_max,
                        eebc_weight=self.eebc_weight
                    )

                    y_pred.append(prediction[0])
                    y_true.append(t_test_i)

                except Exception as e:
                    warnings.warn(f"leave_one_group_out: {e}", UserWarning)

            # Calculate metrics for this fold and save them for later averaging
            evaluation_metrics.append(
                EvaluationMetrics(y_true, y_pred)
            )

        # Average the metrics across all folds
        final_metrics = EvaluationMetrics.average_metrics(evaluation_metrics)

        return final_metrics
