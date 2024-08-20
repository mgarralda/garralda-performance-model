"""
Dou-Performance-Model
Model implementation and the cross-validation techniques for evaluation of the performance

Author: Mariano Garralda-Barrio, Carlos Eiras-Franco, Verónica Bolón-Canedo
Date: 2024

Example Usage Validation:

    validation = CrossValidation(
        # Dataset
        w=prats_workload_descriptors,
        w_groups=w_groups,
        cs=configuration_settings,
        t=target,
        k=k_neighbors
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

import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsTransformer
from sklearn.preprocessing import MinMaxScaler
from evaluation import EvaluationMetrics


class UnsupervisedStage:
    """ Adaptive exploration and exploration of Insightful workload neighbors  """

    def get_optimal_insightful_neighbors(
            self,
            workload_descriptors: np.ndarray,
            workload_ref: np.ndarray,
            k: int,
    ) -> np.array:
        """
        Get optimal number of neighbors

        :param workload_descriptors
        :param workload_ref: reference workload
        :param k: number of neighbors

        :return: k_neighbors_indexes of the nearest neighbors
        """

        # Create a pipeline with a scaler and a KNeighborsTransformer to compute only distances
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('knn_dist', KNeighborsTransformer(n_neighbors=k, metric='euclidean'))
        ])

        pipeline.fit(workload_descriptors)
        knn_dist = pipeline.named_steps['knn_dist']
        workload_ref_scaled = pipeline.named_steps['scaler'].transform(
            # Check if the workload_ref is a 1D array
            workload_ref.reshape(1, -1) if workload_ref.ndim == 1 else workload_ref
        )
        _, indices = knn_dist.kneighbors(workload_ref_scaled, return_distance=True)

        return indices[0]


class SupervisedStage:
    """ Incremental lazy predictive model """

    def get_random_forest_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> GridSearchCV:
        """ Random Forest Regressor model """

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

        regressor.fit(X, y)

        return regressor


class DouPerformanceModel:
    """ Incremental transfer learning adaptation for efficient performance modeling of big data workloads """

    def __init__(self) -> None:
        self.unsupervised_stage = UnsupervisedStage()
        self.supervised_stage = SupervisedStage()

    def predict(
            self,
            workload_descriptors: np.ndarray,
            configuration_settings: np.ndarray,
            execution_times: np.ndarray,
            workload_ref: np.ndarray,
            new_configuration_settings: np.ndarray,
            k: int
    ) -> np.ndarray:
        """
        Instance of the performance model to use in inferencing

        :param workload_descriptors: Prats-descriptors
        :param configuration_settings:
        :param execution_times:
        :param workload_ref:
        :param new_configuration_settings:
        :param k: number of neighbors
        :return: prediction
        """

        k_neighbors_indexes = self.unsupervised_stage.get_optimal_insightful_neighbors(
            workload_descriptors,
            workload_ref,
            k
        )

        regressor = self.supervised_stage.get_random_forest_regressor_model(
            X=configuration_settings[k_neighbors_indexes],
            y=execution_times[k_neighbors_indexes]
        )

        # Check if the new_configuration_settings is a 1D array, thus only sent a single configuration setting to predict
        if new_configuration_settings.ndim == 1:
            new_configuration_settings = new_configuration_settings.reshape(1, -1)

        prediction = regressor.predict(new_configuration_settings)

        return prediction


class CrossValidation:
    """ Cross validation techniques """

    def __init__(
            self,
            w: np.ndarray,
            w_groups: np.ndarray,
            cs: np.ndarray,
            t: np.ndarray,
            k: int,
    ) -> None:
        """ Initialize the cross validation attributes
        Args:
            w (np.ndarray): Garralda workload descriptor.
            w_groups (np.ndarray): Unique groups of workloads.
            cs (np.ndarray): Configuration settings.
            t (np.ndarray): Execution times (target).
            k: number of neighbors
        """

        self.w = w
        self.w_groups = w_groups
        self.cs = cs
        self.t = t
        self.k = k

        self.perf_model = DouPerformanceModel()

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
                prediction = self.perf_model.predict(
                    workload_descriptors=w_train,
                    configuration_settings=cs_train,
                    execution_times=t_train,
                    workload_ref=w_test,
                    new_configuration_settings=cs_test,
                    k=self.k
                )

                # print(f"leave_one_out: {prediction=} | {t_test=}")

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
                    prediction = self.perf_model.predict(
                        workload_descriptors=w_train,
                        configuration_settings=cs_train,
                        execution_times=t_train,
                        workload_ref=w_test_i,
                        new_configuration_settings=cs_test_i,
                        k=self.k
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
