"""
This script performs a Bayesian inversion on the residual depth measurements using a Sobolev prior
defined on the sphere. It reads in real data, constructs the forward model, defines the prior,
and sets up the inversion problem. Additionally, it computes a preconditioner to enhance
the efficiency of the inversion process.
"""
import os
import numpy as np
import pygmt as pg
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import pandas as pd
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev
from time import time
from tqdm import tqdm

pg.config(MAP_FRAME_TYPE="plain")
pg.config(FONT_ANNOT_PRIMARY="10p,Palatino-Roman,black")
pg.config(FONT_ANNOT_SECONDARY="10p,Palatino-Roman,black")
pg.config(FONT_LABEL="10p,Palatino-Roman,black")

inf.configure_threading(n_threads=1)

DT_CMAP = "/space/ij264/earth-tunya/cpts/vik_DT.cpt"
STD_CMAP = "/space/ij264/earth-tunya/cpts/vik_DT_error.cpt"

# --- Configuration ---
DATA_PATH = Path("/space/ij264/earth-tunya/geoinf_analysis/data/global.xyz")
N_DATAS = [10, 100]
LMAX_FULL = 64
LMAX_PRE = 16
MODEL_SPACE_ORDER = 2.0
MODEL_SPACE_SCALE = 0.1
PRIOR_ORDER = 0.1
PRIOR_SCALE = 0.01

for N_DATA in tqdm(N_DATAS):
    FNAME_ROOT = f'/space/ij264/earth-tunya/geoinf_analysis/figures/real_data/{N_DATA}_points/p_order_{PRIOR_ORDER}_s_{PRIOR_SCALE}'
    os.makedirs(FNAME_ROOT, exist_ok=True)

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

    # Form the preconditioner.
    print("Forming the preconditioner...")
    pre_space = Sobolev(LMAX_PRE, MODEL_SPACE_ORDER, MODEL_SPACE_SCALE)
    pre_forward_op = pre_space.point_evaluation_operator(points_to_evaluate_at)
    pre_forward_prob = inf.LinearForwardProblem(
        pre_forward_op, data_error_measure=data_error
    )
    pre_prior = get_constrained_prior(pre_space, PRIOR_ORDER, PRIOR_SCALE)

    pre_inversion = inf.LinearBayesianInversion(pre_forward_prob, pre_prior)
    solver = inf.EigenSolver(parallel=False)
    pre_inversion_normal_op = pre_inversion.normal_operator.extract_diagonal(parallel=True, n_jobs=5)
    pre_inversion_normal_op[pre_inversion_normal_op < 1e-12] = 1.0
    preconditioner = inf.DiagonalSparseMatrixLinearOperator.from_diagonal_values(
        pre_inversion.data_space, pre_inversion.data_space, 1.0 / pre_inversion_normal_op
    )
    # --- Final Inversion ---
    print("Solving the linear system via CG...")
    bi = inf.LinearBayesianInversion(forward_prob, prior_measure)
    posterior_measure = bi.model_posterior_measure(
        data["z"].values,
        inf.CGMatrixSolver(),
        preconditioner=preconditioner,
    )

    # Calculate pointwise estimates of standard deviation.
    print("Sampling from posterior")
    pointwise_variance = posterior_measure.sample_pointwise_variance(
        20, parallel=True, n_jobs=8
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
    N_SAMPLES = 10000
    DEGREES = [2, 5, 10, 20, 30]
    print("Sampling Prior Power...")
    prior_powers = model_space._sample_power_measure(prior_measure, N_SAMPLES, parallel=True, n_jobs=16)

    print("Sampling Posterior Power...")
    posterior_powers = model_space._sample_power_measure(posterior_measure, N_SAMPLES, parallel=True, n_jobs=16)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, layout='constrained')

    # Capture the return to get the 'pc' (mappable) for the colorbar
    _, ax1 = model_space.plot_power_spectrum_2d(prior_powers, ax=ax1)
    # Make sure to return the 'pc' from your function or access it via ax.collections[0]
    pc_last = ax1.collections[0]

    _, ax2 = model_space.plot_power_spectrum_2d(posterior_powers, ax=ax2)

    # Set titles/labels
    ax1.set_title("Prior Power Spectrum")
    ax2.set_title("Posterior Power Spectrum")

    # Add ONE colorbar for the whole figure
    fig.colorbar(ax2.collections[0], ax=[ax1, ax2], label='Sample Density', location='right')

    plt.savefig(f'{FNAME_ROOT}/power_comparison.png', dpi=150)

    # Plot histogram of each degre.
    # for degree in DEGREES:
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     ax.hist(prior_powers[:, degree], bins=30, density=True, alpha=0.7, color='blue')
    #     ax.hist(posterior_powers[:, degree], bins=30, density=True, alpha=0.7, color='orange')
    #     ax.set_title(f'Power Distribution at Degree l={degree}', fontweight='bold')
    #     ax.set_xlabel('Power')
    #     ax.set_ylabel('Counts')
    #     ax.legend(['Prior', 'Posterior'])
    #     plt.savefig(f'{FNAME_ROOT}/power_histogram_l_{degree}.png', dpi=150)
    # 1. Prepare the data into a "Long-Form" DataFrame
    # DEGREES = [2, 8, 30]
    # plot_data = []

    # for l in DEGREES:
    #     # Add Prior samples
    #     for p in prior_powers[:, l]:
    #         plot_data.append({'Degree': f'l={l}', 'Power': p, 'Type': 'Prior'})
    #     # Add Posterior samples
    #     for p in posterior_powers[:, l]:
    #         plot_data.append({'Degree': f'l={l}', 'Power': p, 'Type': 'Posterior'})

    # print(plot_data)
    # df = pd.DataFrame(plot_data)

    # # 2. Plot
    # fig, ax = plt.subplots(figsize=(10, 6))

    # sns.violinplot(data=df, x='Degree', y='Power', hue='Type',
    #                split=True, inner="quart", palette={'Prior': 'lightgrey', 'Posterior': 'orange'},
    #                ax=ax)

    # ax.set_yscale('log')
    # ax.set_title("Power Distribution Comparison (Prior vs Posterior)", fontweight='bold')
    # ax.set_ylabel("Power ($y$-axis)")
    # ax.grid(axis='y', alpha=0.3)

    # plt.savefig(f'{FNAME_ROOT}/power_violin_comparison.png', dpi=150)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define x-positions for each degree
    x_centers = np.arange(len(DEGREES))
    width = 0.3  # Width of the "histogram" column

    for i, l in enumerate(DEGREES):
        # Calculate log-bins for this specific degree
        all_p = np.concatenate([prior_powers[:, l], posterior_powers[:, l]])
        bins = np.logspace(np.log10(all_p.min()), np.log10(all_p.max()), 40)

        # Prior (Left side)
        hist_pr, edges = np.histogram(prior_powers[:, l], bins=bins, density=True)
        # Scale the histogram width to fit in our "bar" slot
        hist_pr = (hist_pr / hist_pr.max()) * width
        ax.barh(edges[:-1], -hist_pr, height=np.diff(edges), left=x_centers[i],
                color='blue', alpha=0.4, label='Prior' if i==0 else "")

        # Posterior (Right side)
        hist_po, edges = np.histogram(posterior_powers[:, l], bins=bins, density=True)
        hist_po = (hist_po / hist_po.max()) * width
        ax.barh(edges[:-1], hist_po, height=np.diff(edges), left=x_centers[i],
                color='orange', alpha=0.7, label='Posterior' if i==0 else "")

    ax.set_yscale('log')
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f'l={l}' for l in DEGREES])
    ax.set_ylabel("Power")
    ax.set_xlabel("Degree / Density")
    ax.legend()
    ax.set_title("Sideways Histograms at Specific Degrees")
    ax.set_ylim(1e-4, 1e1)
    plt.savefig(f'{FNAME_ROOT}/sideways_histograms.png', dpi=150)

    data_list = []

    for l in DEGREES:
        # Add Prior samples
        for p in prior_powers[:, l]:
            data_list.append({'Degree': f'l={l}', 'Power': p, 'Type': 'Prior'})
        # Add Posterior samples
        for p in posterior_powers[:, l]:
            data_list.append({'Type': 'Posterior', 'Degree': f'l={l}', 'Power': p})

    df = pd.DataFrame(data_list)

    fig, ax = plt.subplots(figsize=(10, 6))

    # The 'split=True' creates the half-prior / half-posterior effect
    sns.violinplot(
        data=df,
        x='Degree',
        y='Power',
        hue='Type',
        split=True,
        inner="quart",      # Shows the median and quartiles
        density_norm="count", # Scales the width by number of samples
        ax=ax,
        fill=False,
        bw_adjust=0.5
    )

    # Force the Log Scale and Limits
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1e1)

    ax.set_title("Power Distribution: Prior vs Posterior", fontweight='bold')
    plt.savefig(f'{FNAME_ROOT}/seaborn_power_comparison.png', dpi=150)