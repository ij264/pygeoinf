import numpy as np
import numpy as np
import pygmt as pg
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(7)

REGION = 'd'
PROJECTION = 'H12c'

# Degree of spherical harmonics for test case
LMAX = 128

# Parameters for model space
MODEL_ORDER = 2
MODEL_SCALE = 0.1
RADIUS = 2

PRIOR_ORDER = 4.0
PRIOR_SCALE = 2.0

# Set up model space
model_space = Sobolev(LMAX, MODEL_ORDER, MODEL_SCALE, radius=RADIUS)

# Data point
N_DATA = 1000

DATA_LOCATION = 'global'

raw_points = model_space.random_points(N_DATA)
lats = [lat for lat, _ in raw_points]
lons = [lon for _, lon in raw_points]
STD = .1

pd_coords = pd.DataFrame({
    'longitude': lons,
    'latitude': lats
})

if DATA_LOCATION == 'ocean':
    GRID_MASK_PATH = '/space/ij264/earth-tunya/pygeoinf/dynamic_topography/data/oceanmask.nc'
elif DATA_LOCATION == 'continent':
    GRID_MASK_PATH = '/space/ij264/earth-tunya/pygeoinf/dynamic_topography/data/landmask.nc'
else:
    GRID_MASK_PATH = None

if GRID_MASK_PATH:
    pd_selected_coords = pg.select(data=pd_coords, projection=PROJECTION, gridmask=GRID_MASK_PATH)

    selected_points = list(zip(pd_selected_coords['latitude'], pd_selected_coords['longitude']))
else:
    selected_points = raw_points

# Set up forward operator
forward_operator = model_space.point_evaluation_operator(selected_points)
data_error_measure = inf.GaussianMeasure.from_standard_deviation(forward_operator.codomain, STD)

forward_problem = inf.LinearForwardProblem(
    forward_operator,
    data_error_measure=data_error_measure
)
# Set the unconstrained prior
unconstrained_model_prior_measure = (
    model_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        PRIOR_ORDER, PRIOR_SCALE
    )
)

# Setup Constraint
constraint_operator = model_space.to_coefficient_operator(0, lmin=0)
constraint_value = np.array([0])
constraint = inf.AffineSubspace.from_linear_equation(
    constraint_operator, constraint_value, solver=inf.CholeskySolver()
)

# Form the constrained prior
model_prior_measure = constraint.condition_gaussian_measure(
    unconstrained_model_prior_measure
)

model, data = forward_problem.synthetic_model_and_data(model_prior_measure)


LMAX_TESTs = np.arange(1, 40, 1)
# LMAX_TESTS = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60]
log_evidences = []
for LMAX_TEST in LMAX_TESTs:
    model_space_test = Sobolev(LMAX_TEST, MODEL_ORDER, MODEL_SCALE, radius=RADIUS)

    forward_operator_test = model_space_test.point_evaluation_operator(selected_points)

    forward_problem_test = inf.LinearForwardProblem(
    forward_operator_test,
    data_error_measure=data_error_measure
)
    unconstrained_model_prior_measure = (
        model_space_test.point_value_scaled_sobolev_kernel_gaussian_measure(
            PRIOR_ORDER, PRIOR_SCALE
        )
    )

    constraint_operator = model_space_test.to_coefficient_operator(0, lmin=0)
    constraint_value = np.array([0])
    constraint = inf.AffineSubspace.from_linear_equation(
        constraint_operator, constraint_value, solver=inf.CholeskySolver()
    )

    model_prior_measure = constraint.condition_gaussian_measure(
        unconstrained_model_prior_measure
    )

    evidence_covariance = (
        forward_operator_test @ model_prior_measure.covariance @ forward_operator_test.adjoint
        + data_error_measure.covariance
    )

    evidence_expectation = np.zeros(len(selected_points))

    evidence_covariance_matrix = evidence_covariance.matrix(dense=True)

    evidence_distribution = multivariate_normal(
        mean=evidence_expectation,
        cov=evidence_covariance_matrix,
        allow_singular=True
    )

    log_evidence = evidence_distribution.logpdf(data)
    log_evidences.append(log_evidence)
    print("-" * 30)
    print(f"LMAX: {LMAX_TEST}")
    print(f"Log Evidence: {log_evidence:.4f}")

    inversion = inf.LinearBayesianInversion(forward_problem_test, model_prior_measure)
    model_posterior_measure = inversion.model_posterior_measure(data, inf.CholeskySolver())
    model_posterior_expectation = model_posterior_measure.expectation

    inverted_data = forward_operator_test(model_posterior_expectation)
    print(f"Posterior Model Dimension: {inverted_data.shape}")
    residual = data - forward_operator_test(model_posterior_expectation)
    rmse = np.sqrt(np.mean(residual**2))

    print(f"L: {LMAX_TEST:2d} | Log-Ev: {log_evidence:8.2f} | RMSE: {rmse:8.5f} | Target STD: {STD}")

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(LMAX_TESTs, log_evidences, marker='o')
plt.title('Log Bayesian Evidence vs LMAX')
plt.xlabel('LMAX')
plt.ylabel('Log Bayesian Evidence')
plt.grid()
plt.savefig('/space/ij264/earth-tunya/pygeoinf/figures/log_evidence_vs_lmax.png', dpi=300)