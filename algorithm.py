"""
Garralda-Performance-Model
Adaptive Incremental Transfer Learning for Efficient Performance Modeling of Big Data Workloads

This module implements an adaptive-incremental performance model leveraging both unsupervised and supervised learning stages
to predict execution times for big data workloads, particularly in Apache Spark environments.

Author: Mariano Garralda-Barrio, Carlos Eiras-Franco, Verónica Bolón-Canedo
License: MIT License (see LICENSE file for details)
Date: 2024

Usage:
- Import the `GarraldaPerformanceModel` class and use the `get_performance_model` or `fit_predict` methods to train or inference.
- See the documentation or README for more details on how to use this module.

Example:
    model = GarraldaPerformanceModel()

    # Option 1: To get the trained performance model and predict execution time for new configuration settings
    trained_model = model.train(workload_descriptors, config_settings, exec_times, workload_ref, k_min, k_max)
    predictions = trained_model.predict(new_config_settings)

    # Option 2: To lazy train and predict execution time for new configuration settings in one step
    predictions = model.predict(workload_descriptors, config_settings, exec_times, workload_ref, new_config_settings, k_min, k_max)
"""

import math
import array
import numpy as np
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


class UnsupervisedStage:
    """ Adaptive exploration and exploration of Insightful workload neighbors  """

    def get_optimal_insightful_neighbors(
            self,
            workload_descriptors: np.ndarray,
            execution_times: np.ndarray,
            workload_ref: np.ndarray,
            k_min: int,
            k_max: int,
            eebc_weight: float
    ) -> np.array:
        """
        Get optimal number of neighbors

        :param workload_descriptors
        :param execution_times
        :param workload_ref: reference workload
        :param k_min: minimum number of neighbors
        :param k_max: maximum number of neighbors
        :param eebc_weight: Proportion for the quality and distance

        :return: k_optimal_neighbors_indexes (optimal number of neighbors) of the nearest neighbors
        """

        # Create a pipeline with a scaler and a KNeighborsTransformer to compute only distances
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('knn_dist', KNeighborsTransformer(n_neighbors=k_max, metric='euclidean'))
        ])

        pipeline.fit(workload_descriptors)
        knn_dist = pipeline.named_steps['knn_dist']
        workload_ref_scaled = pipeline.named_steps['scaler'].transform(
            workload_ref.reshape(1, -1) if workload_ref.ndim == 1 else workload_ref
        )
        distances, indices = knn_dist.kneighbors(workload_ref_scaled, return_distance=True)

        nearest_workload = workload_descriptors[indices]
        nearest_targets = execution_times[indices]

        k_min_score = []
        k_max_score = []

        for k in range(k_min, k_max+1):
            w = nearest_workload[0][:k]
            t = nearest_targets[0][:k]
            d = distances[0][:k]

            CqC = self.coefficient_quality_correlation(w, t)
            CdV = self.coefficient_distance_variation(d)

            k_min_score.append(CqC)
            k_max_score.append(CdV)

        k_opt = self.exploration_exploitation_balance_coefficient(
            k_min_score,
            k_max_score,
            eebc_weight
        )

        k_opt += k_min  # k_min-based window index

        k_opt_neighbors_indexes = indices[0][:k_opt]

        return k_opt_neighbors_indexes

    def coefficient_distance_variation(
            self,
            dist: np.ndarray
    ) -> float:
        """
        Coefficient of distance Variation (CdV) is a measure of relative variability. It is the ratio of the standard deviation to the mean.

        :param dist: distances of neighbors relative to the reference workload
        :return: CdV
        """

        std_dist = np.std(dist)
        mean_dist = np.mean(dist)

        if mean_dist == 0:
            return 0
        return std_dist / mean_dist

    def coefficient_quality_correlation(
            self,
            w: np.ndarray,
            t: np.ndarray
    ) -> float:
        """
        Coefficient of quality Correlation (CqC) is a measure of the correlation between two variables.
        The Pearson correlation coefficient measures the linear relationship between two datasets
        The calculation of the p-value relies on the assumption that each dataset is normally distributed.
        Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation.
        Correlations of -1 or +1 imply an exact linear relationship.
        Positive correlations imply that as x increases, so does y. Negative correlations imply that as x increases, y decreases

        :param w: workload_descriptors
        :param t: execution_times:
        :return: CqC
        """

        std_w = np.std(w, axis=1)
        if np.all(std_w == std_w[0]):
            return 0  # Return 0 if the input array is constant

        correlation, _ = pearsonr(std_w, t)

        return abs(correlation)

    def exploration_exploitation_balance_coefficient(
            self,
            q: array,
            d: array,
            weight: float
    ) -> int:
        """
        Exploration-Exploitation Balance Coefficient (EEBC) is a measure of the balance between exploration and exploitation.

        :param q: Coefficient of quality Correlations
        :param d: Coefficient of distance Variations
        :param weight: Proportion for the quality and distance
        :return: EEBC values
        """

        scaler = MinMaxScaler()
        q_scaled = scaler.fit_transform(
            np.array(q).reshape(-1, 1)
        ).flatten()
        d_scaled = scaler.fit_transform(
            np.array(d).reshape(-1, 1)
        ).flatten()

        EEBC = [weight * q_scaled[i] + (1 - weight) * d_scaled[i] for i in range(len(q_scaled))]

        # best exploration_exploitation_balance_coefficient index, thereby the optimal number of neighbors
        k_opt = np.argmax(EEBC)

        return k_opt

    def k_bound_heuristics(
            self,
            dataset_size: int,
            p: float = 0.1
    ) -> Tuple[int, int]:
        """
        Get the bounds of the number of neighbors through the defined heuristics in the paper

        :param dataset_size:
        :param p: percentage of the dataset size
        :return: tuple of the bounds
        """

        n_sqrt = int(math.sqrt(dataset_size))

        k_min = min(3, int(n_sqrt * p))
        k_max = n_sqrt + k_min

        return k_min, k_max


class SupervisedStage:
    """ Incremental lazy predictive model """

    def get_non_negative_least_squares_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> TransformedTargetRegressor:
        """ Non negative least squares regression (NNLS) """

        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            # Positive= True -> It defines a linear regression, with (coefficients) estimation based on non-negative least squares (NNLS)
            ('lr', LinearRegression(positive=True))
        ])

        # Yeo-Johnson technique is applied to the target to mitigate their impact and prevent undue influence from anomalous values.
        # Power transforms, as a family of parametric, monotonic transformations, stabilize variance and make the data more Gaussian-like.
        # This transformation is crucial for addressing issues related to heteroscedasticity (non-constant variance) and other scenarios where
        # normality is desired, which can otherwise violate the assumptions of linear regression.
        regressor = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=PowerTransformer(method="yeo-johnson"),
            check_inverse=False
        )
        regressor.fit(X, y)

        return regressor

    def get_adaboost_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> Pipeline:
        """ AdaBoost regression """

        regressor = Pipeline([
            ('scaler', MinMaxScaler()),
            ('AdaBoost', AdaBoostRegressor())
        ])

        regressor.fit(X, y)

        return regressor

    def get_random_forest_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> Pipeline:
        """ Random Forest regression """

        regressor = Pipeline([
            ('scaler', MinMaxScaler()),
            ('rf', RandomForestRegressor())
        ])

        regressor.fit(X, y)

        return regressor

    def get_gradientboosting_regressor_model(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> Pipeline:
        """ Gradient Boosting regression """

        regressor = Pipeline([
            ('scaler', MinMaxScaler()),
            ('gr', GradientBoostingRegressor())
        ])

        regressor.fit(X, y)

        return regressor


class GarraldaPerformanceModel:
    """ Incremental transfer learning adaptation for efficient performance modeling of big data workloads """

    def __init__(self) -> None:
        self.unsupervised_stage = UnsupervisedStage()
        self.supervised_stage = SupervisedStage()

    def train(
            self,
            workload_descriptors: np.ndarray,
            configuration_settings: np.ndarray,
            execution_times: np.ndarray,
            workload_ref: np.ndarray,
            k_min: int,
            k_max: int,
            eebc_weight: float = 0.5
    ) -> TransformedTargetRegressor:
        """
        Instance of the performance model to use in inferencing

        :param workload_descriptors: Garralda-descriptors
        :param configuration_settings:
        :param execution_times:
        :param workload_ref:
        :param k_min: minimum number of neighbors
        :param k_max: maximum number of neighbors
        :param eebc_weight: Proportion for the quality and distance
        :return: prediction
        """

        k_opt_neighbors_indexes = self.unsupervised_stage.get_optimal_insightful_neighbors(
            workload_descriptors,
            execution_times,
            workload_ref,
            k_min,
            k_max,
            eebc_weight
        )

        regressor = self.supervised_stage.get_non_negative_least_squares_regressor_model(
            X=configuration_settings[k_opt_neighbors_indexes],
            y=execution_times[k_opt_neighbors_indexes]
        )

        """ The other evaluated base-learner regressors and their availability are commented out for reference """
        # regressor = self.supervised_stage.get_random_forest_regressor_model(
        #     X=configuration_settings[k_opt_neighbors_indexes],
        #     y=execution_times[k_opt_neighbors_indexes]
        # )
        # regressor = self.supervised_stage.get_adaboost_regressor_model(
        #     X=configuration_settings[k_opt_neighbors_indexes],
        #     y=execution_times[k_opt_neighbors_indexes]
        # )
        # regressor = self.supervised_stage.get_gradientboosting_regressor_model(
        #     X=configuration_settings[k_opt_neighbors_indexes],
        #     y=execution_times[k_opt_neighbors_indexes]
        # )

        return regressor

    def fit_predict(
            self,
            workload_descriptors: np.ndarray,
            configuration_settings: np.ndarray,
            execution_times: np.ndarray,
            workload_ref: np.ndarray,
            new_configuration_settings: np.ndarray,
            k_min: int,
            k_max: int,
            eebc_weight: float = 0.5
    ) -> np.ndarray:
        """
        Instance of the performance model to use in inferencing

        :param workload_descriptors:
        :param configuration_settings:
        :param execution_times:
        :param workload_ref:
        :param new_configuration_settings: new configuration settings to predict
        :param k_min: minimum number of neighbors
        :param k_max: maximum number of neighbors
        :param eebc_weight: Proportion for the quality and distance
        :return: prediction
        """

        k_opt_neighbors_indexes = self.unsupervised_stage.get_optimal_insightful_neighbors(
            workload_descriptors,
            execution_times,
            workload_ref,
            k_min,
            k_max,
            eebc_weight
        )

        regressor = self.supervised_stage.get_non_negative_least_squares_regressor_model(
            X=configuration_settings[k_opt_neighbors_indexes],
            y=execution_times[k_opt_neighbors_indexes]
        )

        # Check if the new_configuration_settings is a 1D array, thus only sent a single configuration setting to predict
        if new_configuration_settings.ndim == 1:
            new_configuration_settings = new_configuration_settings.reshape(1, -1)

        prediction = regressor.predict(new_configuration_settings)

        return prediction
