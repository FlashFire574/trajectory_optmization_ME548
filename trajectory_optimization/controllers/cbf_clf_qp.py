import functools
from typing import Callable, List, Tuple

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cbfax import (
    ControlBarrierFunction,
    ControlLyapunovFunction,
)
from dynamaxsys import (
    ControlAffineDynamics
)
from matplotlib.patches import Rectangle

class CBFCLFFilter:
    clfs: List[ControlLyapunovFunction]
    cbfs: List[ControlBarrierFunction]
    clf_alphas: List[Callable[float, float]]
    cbf_alphas: List[Callable[float, float]]
    dynamics: ControlAffineDynamics
    control_limits: List[Tuple[float, float]]

    def __init__(
        self,
        clfs: List[ControlLyapunovFunction],
        cbfs: List[ControlBarrierFunction],
        clf_alphas: List[Callable[float, float]],
        cbf_alphas: List[Callable[float, float]],
        dynamics: ControlAffineDynamics,
        control_limits: List[Tuple[float, float]] = None,
        cbf_slack_weighting: List[float] = None,
        clf_slack_weighting: List[float] = None,
    ):
        self.clfs = clfs
        self.cbfs = cbfs
        self.clf_alphas = clf_alphas
        self.cbf_alphas = cbf_alphas
        self.dynamics = dynamics
        self.control_limits = control_limits

        # Assert that all CLFs have the same state_dim and all CBFs have the same state_dim, and that they match
        if self.clfs:
            clf_state_dim = self.clfs[0].state_dim
            assert all(clf.state_dim == clf_state_dim for clf in self.clfs), (
                "All CLFs must have the same state_dim"
            )
        if self.cbfs:
            cbf_state_dim = self.cbfs[0].state_dim
            assert all(cbf.state_dim == cbf_state_dim for cbf in self.cbfs), (
                "All CBFs must have the same state_dim"
            )
        if self.clfs and self.cbfs:
            assert clf_state_dim == cbf_state_dim, (
                "CLFs and CBFs must have the same state_dim"
            )

        assert self.dynamics.state_dim == clf_state_dim, (
            "Dynamics and CLFs must have the same state_dim"
        )

        # Obtain the control dimension, number of CLFs, and number of CBFs
        control_dim = self.dynamics.control_dim
        n_clf = len(self.clfs)
        n_cbf = len(self.cbfs)

        # Set up slack weighting if not provided
        if cbf_slack_weighting is None:
            cbf_slack_weighting = [1000] * len(self.cbfs)
        if clf_slack_weighting is None:
            clf_slack_weighting = [100] * len(self.clfs)


        # Set up CLF filter cvxpy problem
        # - Define the filtered control variable
        self.filtered_control = cp.Variable(control_dim, name="filtered_control")
        # - Define the linear and constant terms for each CLF
        self.linear_clf_terms = [
            cp.Parameter(control_dim, name=f"linear_clf_terms_{i}")
            for i in range(n_clf)
        ]
        self.constant_clf_terms = [
            cp.Parameter(name=f"constant_clf_terms_{i}") for i in range(n_clf)
        ]
        # - Define the slack variables for each CLF
        self.ep_clf = [cp.Variable(1, name=f"ep_clf_{i}") for i in range(n_clf)]
        # - Define the linear and constant terms for each CBF
        self.linear_cbf_terms = [
            cp.Parameter(control_dim, name=f"linear_cbf_terms_{i}")
            for i in range(n_cbf)
        ]
        self.constant_cbf_terms = [
            cp.Parameter(name=f"constant_cbf_terms_{i}") for i in range(n_cbf)
        ]
        # - Define the slack variables for each CBF
        self.ep_cbf = [cp.Variable(1, name=f"ep_cbf_{i}") for i in range(n_cbf)]

        # - Define the desired control input
        self.desired_control = cp.Parameter(control_dim)

        # - Define the slack weighting for each CLF and CBF
        self.clf_slack_weighting = [
            cp.Constant(clf_slack_weighting[i], name=f"clf_slack_weighting_{i}")
            for i in range(n_clf)
        ]
        self.cbf_slack_weighting = [
            cp.Constant(cbf_slack_weighting[i], name=f"cbf_slack_weighting_{i}")
            for i in range(n_cbf)
        ]

        # - Define the objective function
        objective = cp.Minimize(
            # - Minimize the control error
            cp.sum_squares(self.filtered_control - self.desired_control)
            # - Minimize the slack variables
            + sum(
                [
                    self.clf_slack_weighting[i] * cp.sum_squares(self.ep_clf[i])
                    for i in range(n_clf)
                ]
            )
            + sum(
                [
                    self.cbf_slack_weighting[i] * cp.sum_squares(self.ep_cbf[i])
                    for i in range(n_cbf)
                ]
            )
        )

        # - Define the constraints
        constraints = [
            # - CLF constraints
            self.linear_clf_terms[i] @ self.filtered_control
            + self.constant_clf_terms[i]
            <= self.ep_clf[i]
            for i in range(n_clf)
        ]
        constraints += [
            # - CBF constraints
            self.linear_cbf_terms[i] @ self.filtered_control
            + self.constant_cbf_terms[i]
            >= -self.ep_cbf[i]
            for i in range(n_cbf)
        ]
        # - Slack variables must be non-negative
        constraints += [self.ep_clf[i] >= 0 for i in range(n_clf)]
        constraints += [self.ep_cbf[i] >= 0 for i in range(n_cbf)]
        constraints += [
            # - Control limits
            self.filtered_control >= self.control_limits[0],
            self.filtered_control <= self.control_limits[1],
        ]

        # - Construct the cvxpy problem
        self.problem = cp.Problem(objective, constraints)

    def update_cvxpy_parameters(self, state: jnp.ndarray, desired_control: jnp.ndarray, time: float = 0.0):
        """Update the cvxpy problem parameters with the current state and desired control input."""
        # - Update the CLF parameters
        for i, (clf, alpha) in enumerate(zip(self.clfs, self.clf_alphas)):
            linear, constant = clf.control_constraint(state, time)
            self.linear_clf_terms[i].value = np.array(linear)
            self.constant_clf_terms[i].value = np.array(constant)

        # - Update the CBF parameters
        for i, (cbf, alpha) in enumerate(zip(self.cbfs, self.cbf_alphas)):
            linear, constant = cbf.control_constraint(state, time)
            self.linear_cbf_terms[i].value = np.array(linear)
            self.constant_cbf_terms[i].value = np.array(constant)

        # - Update the desired control input
        self.desired_control.value = np.array(desired_control)

    def get_constraint_values(self, state: jnp.ndarray, time: float = 0.0):
        """Get the constraint values for the current state and time."""
        constraint_terms = []
        signs = []
        n_clf = len(self.clfs)
        n_cbf = len(self.cbfs)
        # - Get the CLF constraint values
        for i, (clf, alpha) in enumerate(zip(self.clfs, self.clf_alphas)):
            linear, constant = clf.control_constraint(state, time)
            constraint_terms.append([linear, constant])
            signs.append("<=")

        # - Get the CBF constraint values
        for i, (cbf, alpha) in enumerate(zip(self.cbfs, self.cbf_alphas)):
            linear, constant = cbf.control_constraint(state, time)
            constraint_terms.append([linear, constant])
            signs.append(">=")
        return constraint_terms, signs, n_clf, n_cbf

    def solve(self, verbose=False):
        """Solve the cvxpy problem and return the filtered control input."""
        self.problem.solve(verbose=verbose)
        return self.filtered_control.value
