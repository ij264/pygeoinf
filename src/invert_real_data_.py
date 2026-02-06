"""
This script performs a Bayesian inversion on the residual depth measurements using a Sobolev prior
defined on the sphere. It reads in real data, constructs the forward model, defines the prior,
and sets up the inversion problem. Additionally, it computes a preconditioner to enhance
the efficiency of the inversion process.
"""

import numpy as np
import pygmt as pg
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev

pg.config(MAP_FRAME_TYPE="plain")
pg.config(FONT_ANNOT_PRIMARY="10p,Palatino-Roman,black")
pg.config(FONT_ANNOT_SECONDARY="10p,Palatino-Roman,black")
pg.config(FONT_LABEL="10p,Palatino-Roman,black")

inf.configure_threading(n_threads=1)

DT_CMAP = "/space/ij264/earth-tunya/cpts/vik_DT.cpt"
STD_CMAP = "/space/ij264/earth-tunya/cpts/vik_DT_error.cpt"
FNAME_ROOT = '/space/ij264/earth-tunya/geoinf_analysis/figures/real_data'

# --- Configuration ---
DATA_PATH = Path("/space/ij264/earth-tunya/geoinf_analysis/data/global.xyz")
N_DATA = 1000
LMAX_FULL = 64
LMAX_PRE = 16
MODEL_SPACE_ORDER = 2.0
MODEL_SPACE_SCALE = 0.1
PRIOR_ORDER = 2.0
PRIOR_SCALE = 0.1

# --- Data Loading ---
data = pd.read_csv(
    DATA_PATH,
    names=["lon", "lat", "z", "z_err", "symbol"],
    sep=r"\s+",
).sample(N_DATA)

points_to_evaluate_at = list(zip(data["lat"], data["lon"]))

# --- Helper function for measures ---
def get_constrained_prior(space, order, scale):
    """Encapsulates the creation of a zero-mean Sobolev prior."""
    unconstrained = space.point_value_scaled_sobolev_kernel_gaussian_measure(
        order, scale
    )
    # Zero-mean constraint (l=0)
    constraint_op = space.to_coefficient_operator(0, lmin=0)
    constraint = inf.AffineSubspace.from_linear_equation(
        constraint_op, np.array([0]), solver=inf.CholeskySolver()
    )
    return constraint.condition_gaussian_measure(unconstrained)

model_space = Sobolev(LMAX_FULL, MODEL_SPACE_ORDER, MODEL_SPACE_SCALE)

# Construct forward model.
forward_op = model_space.point_evaluation_operator(points_to_evaluate_at)
data_error = inf.GaussianMeasure.from_standard_deviations(
    forward_op.codomain, data["z_err"].values
)
forward_prob = inf.LinearForwardProblem(forward_op, data_error_measure=data_error)

# Construct prior.
prior_measure = get_constrained_prior(model_space, PRIOR_ORDER, PRIOR_SCALE)

# --- Final Inversion ---
print("Solving the linear system via CG...")
bi = inf.LinearBayesianInversion(forward_prob, prior_measure)
posterior_measure = bi.model_posterior_measure(
    data["z"].values,
    inf.CGMatrixSolver()
)

# Calculate pointwise estimates of standard deviation.
print("Sampling from posterior")
pointwise_variance = posterior_measure.sample_pointwise_variance(
    20, parallel=False
)
pointwise_std = pointwise_variance.copy()
pointwise_std.data = np.sqrt(pointwise_std.data)

# --- Visualization ---
# Plot the posterior mean as well as the standard deviation.
posterior_expectation = posterior_measure.expectation
fig = pg.Figure()
fig.basemap(region='d', projection='H12c', frame='f')
fig.grdimage(grid=posterior_expectation.to_xarray(), cmap=DT_CMAP, dpi=150)
fig.coast(shorelines=True, frame='f', area_thresh=1e5)
fig.colorbar(frame='af+lPosterior Mean Residual Depth (km)', position='+h+e')

fig.shift_origin(xshift='13c')
fig.basemap(region='d', projection='H12c', frame='f')
fig.grdimage(grid=pointwise_std.to_xarray(), cmap=STD_CMAP, dpi=150)
fig.coast(shorelines=True, frame='f', area_thresh=1e5)
fig.colorbar(frame='af+lPosterior Standard Deviation (km)', position='+h+ef', cmap=STD_CMAP)
fig.savefig(f'{FNAME_ROOT}/posterior_mean.png', dpi=150)

# Calculate the power spectrum of the posterior mean and compare to the prior.
print("Sampling Prior Power...")
prior_powers = model_space._sample_power_measure(prior_measure, 100, parallel=True, n_jobs=4)

print("Sampling Posterior Power...")
posterior_powers = model_space._sample_power_measure(posterior_measure, 100, parallel=False)

# 2. Create a Matplotlib figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 3. Plot Prior
# Note: Assuming plot_power_spectrum_2d accepts an 'ax' keyword.
# If it doesn't, we can capture the return and adjust.
model_space.plot_power_spectrum_2d(prior_powers, ax=ax1)
ax1.set_title("Prior Power Spectrum")
ax1.set_xlabel("Spherical Harmonic Degree ($l$)")
ax1.set_ylabel("Power")

# 4. Plot Posterior
model_space.plot_power_spectrum_2d(posterior_powers, ax=ax2)
ax2.set_title("Posterior Power Spectrum")
ax2.set_xlabel("Spherical Harmonic Degree ($l$)")

# 5. Final Polish and Save
plt.tight_layout()
plt.savefig(f'{FNAME_ROOT}/power_comparison.png', dpi=150)