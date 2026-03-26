import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.metrics import (
    rmse_diff_table,
    bias_table,
    sd_table,
    coverage_table,
)


def _pivot_for_heatmap(df: pd.DataFrame, value_col: str, kappa: float) -> pd.DataFrame:
    """
    Filter to one kappa and return a pivot table with:
    rows    = alpha_y
    columns = alpha_d
    values  = value_col
    """
    subset = df[df["kappa"] == kappa].copy()
    if subset.empty:
        raise ValueError(f"No rows found for kappa={kappa}")

    pivot = (
        subset.pivot(index="alpha_y", columns="alpha_d", values=value_col)
        .sort_index()
        .sort_index(axis=1)
    )
    return pivot


def _single_heatmap(
    pivot: pd.DataFrame,
    title: str,
    colorbar_title: str,
    colorscale: str = "RdBu",
    zmid: float | None = None,
    text_round: int = 3,
) -> go.Figure:
    """
    Build a single heatmap from a pivot table.
    """
    z = pivot.values
    x_vals = list(pivot.columns)
    y_vals = list(pivot.index)
    text = np.round(z, text_round)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_vals,
            y=y_vals,
            colorscale=colorscale,
            zmid=zmid,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(title=colorbar_title),
            hovertemplate=(
                "alpha_d=%{x}<br>"
                "alpha_y=%{y}<br>"
                f"{colorbar_title}=%{{z:.4f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis=dict(
            title="Treatment assignment complexity (alpha_d)",
            tickmode="array",
            tickvals=x_vals,
            ticktext=[str(v) for v in x_vals],
        ),
        yaxis=dict(
            title="Outcome nonlinearity (alpha_y)",
            tickmode="array",
            tickvals=y_vals,
            ticktext=[str(v) for v in y_vals],
        ),
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def frontier_heatmap(summary_df: pd.DataFrame, kappa: float) -> go.Figure:
    """
    Heatmap of RMSE_OLS - RMSE_DML for one kappa.
    Positive values mean DML has lower RMSE.
    """
    rmse_df = rmse_diff_table(summary_df)
    pivot = _pivot_for_heatmap(rmse_df, "rmse_diff", kappa)

    fig = _single_heatmap(
        pivot=pivot,
        title=f"Misspecification Frontier: OLS RMSE - DML RMSE (kappa={kappa})",
        colorbar_title="RMSE diff",
        colorscale="RdBu",
        zmid=0,
        text_round=2,
    )
    return fig


def estimator_metric_heatmap(
    summary_df: pd.DataFrame,
    metric: str,
    estimator: str,
    kappa: float,
) -> go.Figure:
    """
    Generic heatmap for one metric and one estimator at a given kappa.
    """
    metric_map = {
        "bias": bias_table,
        "sd": sd_table,
        "coverage": coverage_table,
    }

    if metric not in metric_map:
        raise ValueError(f"Unsupported metric: {metric}")
    if estimator not in {"OLS", "DML"}:
        raise ValueError("Estimator must be 'OLS' or 'DML'")

    wide_df = metric_map[metric](summary_df)
    pivot = _pivot_for_heatmap(wide_df, estimator, kappa)

    title_map = {
        "bias": f"Bias ({estimator}, kappa={kappa})",
        "sd": f"Standard Deviation ({estimator}, kappa={kappa})",
        "coverage": f"Coverage ({estimator}, kappa={kappa})",
    }

    colorbar_map = {
        "bias": f"{estimator} bias",
        "sd": f"{estimator} sd",
        "coverage": f"{estimator} coverage",
    }

    colorscale_map = {
        "bias": "RdBu",
        "sd": "Blues",
        "coverage": "Viridis",
    }

    zmid = 0 if metric == "bias" else None

    fig = _single_heatmap(
        pivot=pivot,
        title=title_map[metric],
        colorbar_title=colorbar_map[metric],
        colorscale=colorscale_map[metric],
        zmid=zmid,
        text_round=2,
    )
    return fig


def frontier_panels(summary_df: pd.DataFrame, kappas: list[float]) -> go.Figure:
    """
    1-row panel plot of frontier heatmaps for multiple kappa values.
    """
    rmse_df = rmse_diff_table(summary_df)

    subplot_titles = [f"kappa={k}" for k in kappas]
    fig = make_subplots(
        rows=1,
        cols=len(kappas),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
    )

    for j, kappa in enumerate(kappas, start=1):
        pivot = _pivot_for_heatmap(rmse_df, "rmse_diff", kappa)
        z = pivot.values
        x_vals = list(pivot.columns)
        y_vals = list(pivot.index)
        text = np.round(z, 2)

        show_scale = j == len(kappas)

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_vals,
                y=y_vals,
                colorscale="RdBu",
                zmid=0,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="RMSE diff") if show_scale else None,
                showscale=show_scale,
                hovertemplate=(
                    "alpha_d=%{x}<br>"
                    "alpha_y=%{y}<br>"
                    "RMSE diff=%{z:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=j,
        )

        fig.update_xaxes(
            title_text="Treatment complexity (α_D)",
            tickmode="array",
            tickvals=x_vals,
            ticktext=[str(v) for v in x_vals],
            row=1,
            col=j,
        )
        fig.update_yaxes(
            title_text="Treatment complexity (α_D)" if j == 1 else None,
            tickmode="array",
            tickvals=y_vals,
            ticktext=[str(v) for v in y_vals],
            scaleanchor=f"x{j}" if j > 1 else "x",
            scaleratio=1,
            row=1,
            col=j,
        )

    fig.update_layout(
        title={"text": "RMSE Difference (OLS − DML) Across Overlap Regimes", "x": 0.5},
        height=450,
        width=350 * len(kappas),
    )

    return fig


def metric_panels(
    summary_df: pd.DataFrame,
    metric: str,
    kappas: list[float],
    estimator: str | None = None,
    text_round: int = 2,
) -> go.Figure:
    """
    Generic 1-row panel heatmap across kappas.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Scenario-level summary dataframe.
    metric : str
        One of {"rmse_diff", "bias", "sd", "coverage"}.
    kappas : list[float]
        Values of kappa to panel across.
    estimator : str | None
        Required for {"bias", "sd", "coverage"} and must be one of {"OLS", "DML"}.
        Must be None for "rmse_diff".
    text_round : int
        Number of decimals shown in cells.
    """
    metric_builders = {
        "rmse_diff": rmse_diff_table,
        "bias": bias_table,
        "sd": sd_table,
        "coverage": coverage_table,
    }

    if metric not in metric_builders:
        raise ValueError("metric must be one of {'rmse_diff', 'bias', 'sd', 'coverage'}")

    if metric == "rmse_diff":
        if estimator is not None:
            raise ValueError("estimator must be None when metric='rmse_diff'")
        metric_df = metric_builders[metric](summary_df)
        value_col = "rmse_diff"
    else:
        if estimator not in {"OLS", "DML"}:
            raise ValueError("For bias/sd/coverage, estimator must be 'OLS' or 'DML'")
        metric_df = metric_builders[metric](summary_df)
        value_col = estimator

    title_map = {
        "rmse_diff": "RMSE Difference (OLS − DML) Across Overlap Regimes",
        "bias": f"Bias ({estimator}) Across Overlap Regimes",
        "sd": f"Standard Deviation ({estimator}) Across Overlap Regimes",
        "coverage": f"Coverage ({estimator}) Across Overlap Regimes",
    }

    colorbar_map = {
        "rmse_diff": "RMSE diff",
        "bias": f"{estimator} bias",
        "sd": f"{estimator} sd",
        "coverage": f"{estimator} coverage",
    }

    colorscale_map = {
        "rmse_diff": "RdBu",
        "bias": "RdBu",
        "sd": "Blues",
        "coverage": "Viridis",
    }

    zmid_map = {
        "rmse_diff": 0,
        "bias": 0,
        "sd": None,
        "coverage": None,
    }

    subplot_titles = [f"kappa={k}" for k in kappas]

    fig = make_subplots(
        rows=1,
        cols=len(kappas),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
    )

    for j, kappa in enumerate(kappas, start=1):
        pivot = _pivot_for_heatmap(metric_df, value_col, kappa)

        z = pivot.values
        x_vals = list(pivot.columns)
        y_vals = list(pivot.index)
        text = np.round(z, text_round)

        show_scale = j == len(kappas)

        hover_name = colorbar_map[metric]

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_vals,
                y=y_vals,
                colorscale=colorscale_map[metric],
                zmid=zmid_map[metric],
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title=colorbar_map[metric]) if show_scale else None,
                showscale=show_scale,
                hovertemplate=(
                    "alpha_d=%{x}<br>"
                    "alpha_y=%{y}<br>"
                    f"{hover_name}=%{{z:.4f}}<extra></extra>"
                ),
            ),
            row=1,
            col=j,
        )

        fig.update_xaxes(
            title_text="Treatment assignment complexity (alpha_d)",
            tickmode="array",
            tickvals=x_vals,
            ticktext=[str(v) for v in x_vals],
            row=1,
            col=j,
        )

        fig.update_yaxes(
            title_text="Outcome nonlinearity (alpha_y)" if j == 1 else None,
            tickmode="array",
            tickvals=y_vals,
            ticktext=[str(v) for v in y_vals],
            scaleanchor=f"x{j}" if j > 1 else "x",
            scaleratio=1,
            row=1,
            col=j,
        )

    fig.update_layout(
        title={"text": title_map[metric], "x": 0.5},
        height=450,
        width=500 * len(kappas),
    )

    return fig


def estimator_metric_panels(
    summary_df: pd.DataFrame,
    metric: str,
    estimator: str,
    kappas: list[float],
) -> go.Figure:
    return metric_panels(
        summary_df=summary_df,
        metric=metric,
        estimator=estimator,
        kappas=kappas,
    )

def combined_theory_empirical_frontier(
    rmse_df: pd.DataFrame,
    kappa_values=(0.5, 1.0, 2.0),
    alpha_y_grid=np.linspace(0, 1, 101),
    alpha_d_grid=np.linspace(0, 1, 101),
    a=1.2,   # benefit from outcome nonlinearity
    b=0.25,  # smaller benefit from treatment learnability
    c=0.55,  # overlap penalty
    d=0.45   # offset
):
    """
    Create a 2xK panel figure:
      - top row: theoretical contour plots
      - bottom row: empirical contour plots based on rmse_diff

    Theoretical score:
        S(alpha_y, alpha_d; kappa) = a*alpha_y + b*alpha_d - c*kappa^2 - d

    Interpretation:
        S > 0  => DML better
        S < 0  => OLS better
    """

    kappa_values = list(kappa_values)
    ncols = len(kappa_values)

    fig = make_subplots(
        rows=2,
        cols=ncols,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
        subplot_titles=(
            [f"Theory: kappa = {k}" for k in kappa_values] +
            [f"Empirical: kappa = {k}" for k in kappa_values]
        )
    )

    # ---------- Top row: theoretical ----------
    AY, AD = np.meshgrid(alpha_y_grid, alpha_d_grid, indexing="ij")

    theory_scores = {}
    theory_absmax = 0.0

    for kappa in kappa_values:
        S = a * AY + b * AD - c * (kappa ** 2) - d
        theory_scores[kappa] = S
        theory_absmax = max(theory_absmax, np.max(np.abs(S)))

    for col, kappa in enumerate(kappa_values, start=1):
        S = theory_scores[kappa]

        fig.add_trace(
            go.Contour(
                x=alpha_d_grid,
                y=alpha_y_grid,
                z=S,
                colorscale="RdBu",
                zmid=0,
                zmin=-theory_absmax,
                zmax=theory_absmax,
                showscale=(col == ncols),
                colorbar=dict(
                    title="Theory score",
                    x=1.02,
                    y=0.80,
                    len=0.35
                ) if col == ncols else None,
                contours=dict(
                    start=-theory_absmax,
                    end=theory_absmax,
                    size=theory_absmax / 12 if theory_absmax > 0 else 0.1
                ),
                line=dict(width=0.6),
                hovertemplate=(
                    "alpha_d=%{x:.2f}<br>"
                    "alpha_y=%{y:.2f}<br>"
                    f"kappa={kappa:.1f}<br>"
                    "score=%{z:.3f}<extra></extra>"
                )
            ),
            row=1, col=col
        )

        # Zero contour
        fig.add_trace(
            go.Contour(
                x=alpha_d_grid,
                y=alpha_y_grid,
                z=S,
                contours=dict(
                    start=0,
                    end=0,
                    coloring="lines"
                ),
                line=dict(width=4, color="gold"),
                showscale=False,
                hoverinfo="skip"
            ),
            row=1, col=col
        )

    # ---------- Bottom row: empirical ----------
    emp_absmax = max(abs(rmse_df["rmse_diff"].min()), abs(rmse_df["rmse_diff"].max()))

    for col, kappa in enumerate(kappa_values, start=1):
        sub = rmse_df[rmse_df["kappa"] == kappa].copy()

        pivot = (
            sub.pivot(index="alpha_y", columns="alpha_d", values="rmse_diff")
            .sort_index()
            .sort_index(axis=1)
        )

        x = pivot.columns.to_numpy()
        y = pivot.index.to_numpy()
        z = pivot.to_numpy()

        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=z,
                colorscale="RdBu",
                zmid=0,
                zmin=-emp_absmax,
                zmax=emp_absmax,
                showscale=(col == ncols),
                colorbar=dict(
                    title="RMSE diff",
                    x=1.02,
                    y=0.20,
                    len=0.35
                ) if col == ncols else None,
                contours=dict(
                    start=-emp_absmax,
                    end=emp_absmax,
                    size=emp_absmax / 12 if emp_absmax > 0 else 0.01
                ),
                line=dict(width=0.6),
                hovertemplate=(
                    "alpha_d=%{x:.2f}<br>"
                    "alpha_y=%{y:.2f}<br>"
                    f"kappa={kappa:.1f}<br>"
                    "rmse_diff=%{z:.4f}<extra></extra>"
                )
            ),
            row=2, col=col
        )

        # Zero contour
        if np.nanmin(z) <= 0 <= np.nanmax(z):
            fig.add_trace(
                go.Contour(
                    x=x,
                    y=y,
                    z=z,
                    contours=dict(
                        start=0,
                        end=0,
                        coloring="lines"
                    ),
                    line=dict(width=4, color="gold"),
                    showscale=False,
                    hoverinfo="skip"
                ),
                row=2, col=col
            )

    # ---------- Axes ----------
    for col in range(1, ncols + 1):
        fig.update_xaxes(title_text="Treatment complexity (alpha_d)", row=1, col=col)
        fig.update_xaxes(title_text="Treatment complexity (alpha_d)", row=2, col=col)
        fig.update_yaxes(title_text="Outcome nonlinearity (alpha_y)", row=1, col=col)
        fig.update_yaxes(title_text="Outcome nonlinearity (alpha_y)", row=2, col=col)

    fig.update_layout(
        title="Theoretical and empirical frontiers for DML vs OLS",
        template="plotly_white",
        width=1600,
        height=950,
        margin=dict(t=90, l=60, r=120, b=60)
    )

    return fig