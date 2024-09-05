"""
Prats-Performance-Model
Model implementation and the cross-validation techniques for evaluation of the performance

Author: Mariano Garralda-Barrio, Carlos Eiras-Franco, Verónica Bolón-Canedo
Date: 2024

Example Usage Validation:

    validation = CrossValidation(
        # Dataset
        w=prats_workload_descriptors,
        w_groups=w_groups,
        ids=input_data_sizes,
        cs=configuration_settings,
        t=target
    )

    loocv_eval = validation.leave_one_out()
    logocv_eval = validation.leave_one_group_out()

    print(f"LOOCV: {loocv_eval}")
    print(f"LOGOCV: {logocv_eval}")

    ame = EvaluationMetrics.AME(
        loocv_eval.HME(),
        logocv_eval.HME()
    )

    print(f"AME: {ame:.2f}")
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut, LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from evaluation import EvaluationMetrics


class PratsPerformanceModel:
    """ Incremental transfer learning adaptation for efficient performance modeling of big data workloads """

    def get_random_forest_regressor_model(
            self,
            workload_descriptors: np.ndarray,
            input_data_sizes: np.ndarray,
            configuration_settings: np.ndarray,
            execution_times: np.ndarray,
    ) -> GridSearchCV:
        """
        Random Forest Regressor model for performance prediction

        :param workload_descriptors: Prats-descriptors
        :param input_data_sizes: size of the data
        :param configuration_settings: configuration settings
        :param execution_times

        :return: performance model
        """

        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('rf', RandomForestRegressor())
        ])

        # Parameter grid for RandomForest based on the original paper
        parameters = {
            'rf__n_estimators': [10, 20, 50, 100, 200],
            'rf__max_depth': [10, 20, None],
            'rf__max_features': ['auto', 'sqrt']
        }

        regressor = GridSearchCV(pipeline, parameters)

        regressor.fit(
            np.concatenate(
                (
                    workload_descriptors,
                    input_data_sizes,
                    configuration_settings
                ),
                axis=1
            ),
            execution_times
        )

        return regressor


class CrossValidation:

    def __init__(
            self,
            w: np.ndarray,
            w_groups: np.ndarray,
            ids: np.ndarray,
            cs: np.ndarray,
            t: np.ndarray,
    ) -> None:
        """ Initialize the cross validation attributes
        Args:
            w (np.ndarray): workload descriptor.
            w_groups (np.ndarray): Unique groups of workloads.
            ids(np.ndarray): Input data sizes.
            cs (np.ndarray): Configuration settings.
            t (np.ndarray): Execution times (target).
        """

        self.w = w
        self.w_groups = w_groups
        self.cs = cs
        self.ids = ids
        self.t = t

        self.perf_model = PratsPerformanceModel()

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
            ids_train, ids_test = self.ids[train_index], self.ids[test_index]
            cs_train, cs_test = self.cs[train_index], self.cs[test_index]
            t_train, t_test = self.t[train_index], self.t[test_index]

            trained_model = self.perf_model.get_random_forest_regressor_model(
                workload_descriptors=w_train,
                input_data_sizes=ids_train,
                configuration_settings=cs_train,
                execution_times=t_train,
            )
            prediction = trained_model.predict(
                np.concatenate(
                    (
                        w_test,
                        ids_test,
                        cs_test
                    ),
                    axis=1
                )
            )

            y_pred.append(prediction)
            y_true.append(t_test)
  
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
            ids_train, ids_test = self.ids[train_index], self.ids[test_index]
            cs_train, cs_test = self.cs[train_index], self.cs[test_index]
            t_train, t_test = self.t[train_index], self.t[test_index]

            try:
                trained_model = self.perf_model.get_random_forest_regressor_model(
                    X=w_train,
                    y=t_train
                )

                # For each sample in test_data (workload group) predict the execution time
                for i in range(len(w_test)):
                    prediction = trained_model.predict([
                        np.concatenate(
                            (
                                w_test,
                                ids_test,
                                cs_test
                            ),
                            axis=1
                        )
                    ])
                  
                    y_pred.append(prediction[0])
                    y_true.append(t_test[i])

            except Exception as e:
                warnings.warn(f"leave_one_group_out: {e}", UserWarning)

            # Calculate metrics for this fold and save them for later averaging
            evaluation_metrics.append(
                EvaluationMetrics(y_true, y_pred)
            )

        # Average the metrics across all folds
        final_metrics = EvaluationMetrics.average_metrics(evaluation_metrics)

        return final_metrics
