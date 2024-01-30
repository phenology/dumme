"""
Mixed Effects Model.
"""

import logging
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin, check_array, check_is_fitted
from sklearn.dummy import DummyRegressor
from sklearn.utils import check_X_y

logger = logging.getLogger(__name__)


# RegressorMixin adds attribute `_estimator_type = "regressor` and `score` method
# BaseEstimator has boilerplate for things like get_params, set_params, _validate_data
class MixedEffectsModel(RegressorMixin, BaseEstimator):
    """
    Scikit-learn compatbile implementation of a mixed-effects model of the form

    .. math::

        y = f(X) + b_i Z + e

    * y is the target variable. The current code only supports regression for
      now, e.g. continuously varying scalar value
    * X is the fixed effect features. Assume p dimensional
    * f(.) is the nonlinear fixed effects mode, e.g. random forest
    * Z is the random effect features. Assume q dimensional.
    * e is iid noise ~N(0, sigma_e²)
    * i is the cluster index. Assume k clusters in the training.
    * bi is the random effect coefficients. They are different per cluster i but
      are assumed to be drawn from the same distribution ~N(0, Sigma_b) where
      Sigma_b is learned from the data.

    Args:
        gll_early_stop_threshold (float): early stopping threshold on GLL
        improvement max_iterations (int): maximum number of EM iterations
    """

    def __init__(
        self,
        fe_model=None,
        gll_early_stop_threshold=None,
        max_iterations=20,
    ):
        self.fe_model = fe_model
        self.gll_early_stop_threshold = gll_early_stop_threshold
        self.max_iterations = max_iterations

    def predict(self, X: ArrayLike):
        """
        Predict using trained MERF.  For known clusters the trained random
        effect correction is applied. For unknown clusters the pure fixed effect
        (RF) estimate is used.

        Note that the shape of X, including ordering of the columns, should be
        the same as what was used for the `fit` method.

        Args:
            X: predictors (both fixed and random effect covariates)

        Returns:
            np.ndarray: the predictions y_hat
        """
        check_is_fitted(self, attributes=["trained_fe_model_", "n_features_in_"])
        X = check_array(X)

        X, clusters, Z = self._split_X_input(X)
        Z = np.array(
            Z
        )  # cast Z to numpy array (required if it's a dataframe, otw, the matrix mults later fail)

        # Apply fixed effects model to all
        y_hat = self.trained_fe_model_.predict(X)

        # Apply random effects correction to all known clusters. Note that then, by default, the new clusters get no
        # random effects correction -- which is the desired behavior.
        for cluster_id in self.cluster_counts_.index:
            indices_i = clusters == cluster_id

            # If cluster doesn't exist in test data that's ok. Just move on.
            if len(indices_i) == 0:
                continue

            # If cluster does exist, apply the correction.
            b_i = self.trained_b_.loc[cluster_id]
            Z_i = Z[indices_i]
            y_hat[indices_i] += Z_i.dot(b_i)

        return y_hat

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        cluster_column: Union[int, str] = -1,
        fixed_effects: Union[int, str, list[int], list[str]] = [],
        random_effects: Union[int, str, list[int], list[str]] = [],
        X_val: ArrayLike = None,
        y_val: ArrayLike = None,
    ):
        """
        Fit MERF using Expectation-Maximization algorithm.

        Args:
            X: predictors (both fixed and random effect covariates)
            y: response/target variable
            cluster_column: name or index of column in X that contains cluster
                assignments. Default is last column.
            fixed_effects: columns (names or indices) to use as fixed effects.
                If not specified, all columns except for those designated as cluster
                or random effects are used.
            random_effects: columns (names or indices) to use as random effects.
                If not specified, an array of ones with shape (n, 1) is used,
                where n = len(X)
            X_val: validation array. If passed, validation loss against
                the validation set is logged during model training.
            y_val: validation array. If passed, validation loss against
                the validation set is logged during model training.

        Returns:
            MERF: fitted model
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Parse Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        check_X_y(
            X, y, ensure_min_features=2
        )  # First check, then parse columns, then overwrite X and y
        self._parse_fit_kwargs(X, cluster_column, fixed_effects, random_effects)
        X, y = check_X_y(X, y, y_numeric=True)

        self.n_features_in_ = X.shape[1]

        # Now split the input into fixed/random effects and cluster assignment
        X, clusters, Z = self._split_X_input(X)

        self.cluster_counts_ = None
        if self.fe_model is None:
            self.trained_fe_model_ = DummyRegressor()
        else:
            self.trained_fe_model_ = clone(self.fe_model)
        self.trained_b_ = None

        self.b_hat_history_ = []
        self.sigma2_hat_history_ = []
        self.D_hat_history_ = []
        self.gll_history_ = []
        self.val_loss_history_ = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        n_clusters = clusters.nunique()
        n_obs = len(y)
        q = Z.shape[1]  # random effects dimension
        Z = np.array(
            Z
        )  # cast Z to numpy array (required if it's a dataframe, otw, the matrix mults later fail)

        logger.warning(f"{n_clusters=}, performance scales with number of clusters.")

        # Create a series where cluster_id is the index and n_i is the value
        self.cluster_counts_ = clusters.value_counts()

        # Do expensive slicing operations only once
        Z_by_cluster = {}
        y_by_cluster = {}
        n_by_cluster = {}
        I_by_cluster = {}
        indices_by_cluster = {}

        # TODO: Can these be replaced with groupbys? Groupbys are less understandable than brute force.
        for cluster_id in self.cluster_counts_.index:
            # Find the index for all the samples from this cluster in the large vector
            indices_i = clusters == cluster_id
            indices_by_cluster[cluster_id] = indices_i

            # Slice those samples from Z and y
            Z_by_cluster[cluster_id] = Z[indices_i]
            y_by_cluster[cluster_id] = y[indices_i]

            # Get the counts for each cluster and create the appropriately sized identity matrix for later computations
            n_by_cluster[cluster_id] = self.cluster_counts_[cluster_id]
            I_by_cluster[cluster_id] = np.eye(self.cluster_counts_[cluster_id])

        # Intialize for EM algorithm
        iteration = 0
        # Note we are using a dataframe to hold the b_hat because this is easier to index into by cluster_id
        # Before we were using a simple numpy array -- but we were indexing into that wrong because the cluster_ids
        # are not necessarily in order.
        b_hat_df = pd.DataFrame(
            np.zeros((n_clusters, q)), index=self.cluster_counts_.index
        )
        sigma2_hat = 1
        D_hat = np.eye(q)

        # vectors to hold history
        self.b_hat_history_.append(b_hat_df)
        self.sigma2_hat_history_.append(sigma2_hat)
        self.D_hat_history_.append(D_hat)

        early_stop_flag = False

        while iteration < self.max_iterations and not early_stop_flag:
            iteration += 1
            logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            logger.debug("Iteration: {}".format(iteration))
            logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ E-step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # fill up y_star for all clusters
            y_star = np.zeros(len(y))
            for cluster_id in self.cluster_counts_.index:
                # Get cached cluster slices
                y_i = y_by_cluster[cluster_id]
                Z_i = Z_by_cluster[cluster_id]
                b_hat_i = b_hat_df.loc[cluster_id]  # used to be ix
                logger.debug(
                    "E-step, cluster {}, b_hat = {}".format(cluster_id, b_hat_i)
                )
                indices_i = indices_by_cluster[cluster_id]

                # Compute y_star for this cluster and put back in right place
                y_star_i = y_i - Z_i.dot(b_hat_i)
                y_star[indices_i] = y_star_i

            # check that still one dimensional
            # TODO: Other checks we want to do?
            assert len(y_star.shape) == 1

            # Do the fixed effects regression with all the fixed effects features
            self.trained_fe_model_ = self.trained_fe_model_.fit(X, y_star)
            f_hat = self.trained_fe_model_.predict(X)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ M-step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            sigma2_hat_sum = 0
            D_hat_sum = 0

            for cluster_id in self.cluster_counts_.index:
                # Get cached cluster slices
                indices_i = indices_by_cluster[cluster_id]
                y_i = y_by_cluster[cluster_id]
                Z_i = Z_by_cluster[cluster_id]
                n_i = n_by_cluster[cluster_id]
                I_i = I_by_cluster[cluster_id]

                # index into f_hat
                f_hat_i = f_hat[indices_i]

                # Compute V_hat_i
                V_hat_i = Z_i.dot(D_hat).dot(Z_i.T) + sigma2_hat * I_i

                # Compute b_hat_i
                V_hat_inv_i = np.linalg.pinv(V_hat_i)
                logger.debug(
                    "M-step, pre-update, cluster {}, b_hat = {}".format(
                        cluster_id, b_hat_df.loc[cluster_id]
                    )
                )
                b_hat_i = D_hat.dot(Z_i.T).dot(V_hat_inv_i).dot(y_i - f_hat_i)
                logger.debug(
                    "M-step, post-update, cluster {}, b_hat = {}".format(
                        cluster_id, b_hat_i
                    )
                )

                # Compute the total error for this cluster
                eps_hat_i = y_i - f_hat_i - Z_i.dot(b_hat_i)

                logger.debug("------------------------------------------")
                logger.debug("M-step, cluster {}".format(cluster_id))
                logger.debug(
                    "error squared for cluster = {}".format(eps_hat_i.T.dot(eps_hat_i))
                )

                # Store b_hat for cluster both in numpy array and in dataframe
                # Note this HAS to be assigned with loc, otw whole df get erroneously assigned and things go to hell
                b_hat_df.loc[cluster_id, :] = b_hat_i
                logger.debug(
                    "M-step, post-update, recalled from db, cluster {}, "
                    "b_hat = {}".format(cluster_id, b_hat_df.loc[cluster_id])
                )

                # Update the sums for sigma2_hat and D_hat. We will update after the entire loop over clusters
                sigma2_hat_sum += eps_hat_i.T.dot(eps_hat_i) + sigma2_hat * (
                    n_i - sigma2_hat * np.trace(V_hat_inv_i)
                )
                D_hat_sum += np.outer(b_hat_i, b_hat_i) + (
                    D_hat - D_hat.dot(Z_i.T).dot(V_hat_inv_i).dot(Z_i).dot(D_hat)
                )  # noqa: E127

            # Normalize the sums to get sigma2_hat and D_hat
            sigma2_hat = (1.0 / n_obs) * sigma2_hat_sum
            D_hat = (1.0 / n_clusters) * D_hat_sum

            logger.debug("b_hat = {}".format(b_hat_df))
            logger.debug("sigma2_hat = {}".format(sigma2_hat))
            logger.debug("D_hat = {}".format(D_hat))

            # Store off history so that we can see the evolution of the EM algorithm
            self.b_hat_history_.append(b_hat_df.copy())
            self.sigma2_hat_history_.append(sigma2_hat)
            self.D_hat_history_.append(D_hat)

            # Generalized Log Likelihood computation to check convergence
            gll = 0
            for cluster_id in self.cluster_counts_.index:
                # Get cached cluster slices
                indices_i = indices_by_cluster[cluster_id]
                y_i = y_by_cluster[cluster_id]
                Z_i = Z_by_cluster[cluster_id]
                I_i = I_by_cluster[cluster_id]

                # Slice f_hat and get b_hat
                f_hat_i = f_hat[indices_i]
                R_hat_i = sigma2_hat * I_i
                b_hat_i = b_hat_df.loc[cluster_id]

                # Numerically stable way of computing log(det(A))
                _, logdet_D_hat = np.linalg.slogdet(D_hat)
                _, logdet_R_hat_i = np.linalg.slogdet(R_hat_i)

                gll += (
                    (y_i - f_hat_i - Z_i.dot(b_hat_i))
                    .T.dot(np.linalg.pinv(R_hat_i))
                    .dot(y_i - f_hat_i - Z_i.dot(b_hat_i))
                    + b_hat_i.T.dot(np.linalg.pinv(D_hat)).dot(b_hat_i)
                    + logdet_D_hat
                    + logdet_R_hat_i
                )  # noqa: E127

            logger.info("Training GLL is {} at iteration {}.".format(gll, iteration))
            self.gll_history_.append(gll)

            # Save off the most updated random effects coefficents
            self.trained_b_ = b_hat_df

            # Early Stopping. This code is entered only if the early stop threshold is specified and
            # if the gll_history array is longer than 1 element, e.g. we are past the first iteration.
            if self.gll_early_stop_threshold is not None and len(self.gll_history_) > 1:
                curr_threshold = np.abs(
                    (gll - self.gll_history_[-2]) / self.gll_history_[-2]
                )
                logger.debug("stop threshold = {}".format(curr_threshold))

                if curr_threshold < self.gll_early_stop_threshold:
                    logger.info(
                        "Gll {} less than threshold {}, stopping early ...".format(
                            gll, curr_threshold
                        )
                    )
                    early_stop_flag = True

            # Compute Validation Loss
            if X_val is not None:
                yhat_val = self.predict(X_val)
                val_loss = np.square(np.subtract(y_val, yhat_val)).mean()
                logger.info(
                    f"Validation MSE Loss is {val_loss} at iteration {iteration}."
                )
                self.val_loss_history_.append(val_loss)

        return self

    def get_bhat_history_df(self):
        """
        This function does a complicated reshape and re-indexing operation to get the
        list of dataframes for the b_hat_history into a multi-indexed dataframe.  This
        dataframe is easier to work with in plotting utilities and other downstream
        analyses than the list of dataframes b_hat_history.

        Args:
            b_hat_history (list): list of dataframes of bhat at every iteration

        Returns:
            pd.DataFrame: multi-index dataframe with outer index as iteration, inner index as cluster
        """
        # Step 1 - vertical stack all the arrays at each iteration into a single numpy array
        b_array = np.vstack(self.b_hat_history_)

        # Step 2 - Create the multi-index. Note the outer index is iteration. The inner index is cluster.
        iterations = range(len(self.b_hat_history_))
        clusters = self.b_hat_history_[0].index
        mi = pd.MultiIndex.from_product(
            [iterations, clusters], names=("iteration", "cluster")
        )

        # Step 3 - Create the multi-indexed dataframe
        b_hat_history_df = pd.DataFrame(b_array, index=mi)
        return b_hat_history_df

    def _parse_fit_kwargs(
        self,
        X,
        cluster_column: Union[int, str] = -1,
        fixed_effects: Union[int, str, list[int], list[str]] = [],
        random_effects: Union[int, str, list[int], list[str]] = [],
    ):
        """Store column indices for fixed and random effects, and clusters."""
        if not isinstance(random_effects, list):
            random_effects = [random_effects]
        if not isinstance(fixed_effects, list):
            fixed_effects = [fixed_effects]

        if isinstance(X, pd.DataFrame):
            if isinstance(cluster_column, str):
                cluster_column = X.columns.get_loc(cluster_column)

            if random_effects:
                if all([isinstance(name, str) for name in random_effects]):
                    random_effects = [
                        X.columns.get_loc(name) for name in random_effects
                    ]

                elif not all([isinstance(item, int) for item in random_effects]):
                    raise ValueError(
                        "Got mixed input for random_effects."
                        "Provide a list of only integers or only strings."
                    )

            if fixed_effects:
                if all([isinstance(name, str) for name in fixed_effects]):
                    fixed_effects = [X.columns.get_loc(name) for name in fixed_effects]

                elif not all([isinstance(item, int) for item in fixed_effects]):
                    raise ValueError(
                        "Got mixed input for fixed_effects."
                        "Provide a list of only integers or only strings."
                    )

        # Now everything should be integers
        if (
            not isinstance(cluster_column, int)
            or not np.all([isinstance(item, int) for item in random_effects])
            or not np.all([isinstance(item, int) for item in fixed_effects])
        ):
            raise ValueError(
                """
                Unable to parse inputs for fit. If X is a numpy array, make sure
                column indices are passed as ints. If X is a pandas array,
                column indices may be str or int, but not mixed.
            """
            )

        ncols = np.asarray(X).shape[1]
        cluster_column %= ncols  # that way -1 also works

        if not fixed_effects:
            fixed_effects = [
                i
                for i in range(ncols)
                if i not in random_effects and i != cluster_column
            ]

        self.cluster_column_ = cluster_column
        self.random_effects_ = random_effects
        self.fixed_effects_ = fixed_effects

    def _split_X_input(self, X):
        """Divide array into fixed and random effects, and clusters."""
        clusters = pd.Series(X[:, self.cluster_column_])
        Z = X[:, self.random_effects_] if self.random_effects_ else np.ones((len(X), 1))
        X_ = X[:, self.fixed_effects_]
        return X_, clusters, Z
