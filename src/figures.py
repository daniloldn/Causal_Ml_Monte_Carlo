import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.metrics import (
    rmse_diff_table,
    bias_table,
    sd_table,
    coverage_table,
    res_var_table, 
    overlap_table
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
                #text=text,
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
        "overlap": overlap_table,
        "residual": res_var_table
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
        "overlap": f"Overlap Across Overlap Regimes",
        "residual": f"Residualized D Variance({estimator}) Across Overlap Regimes"
    }

    colorbar_map = {
        "rmse_diff": "RMSE diff",
        "bias": f"{estimator} bias",
        "sd": f"{estimator} sd",
        "coverage": f"{estimator} coverage",
        "overlap": f"{estimator} coverage",
        "residual": f"{estimator} coverage",
    }

    colorscale_map = {
        "rmse_diff": "RdBu",
        "bias": "RdBu",
        "sd": "Blues",
        "coverage": "Viridis",
        "overlap": "Viridis",
        "residual": "Viridis"
    }

    zmid_map = {
        "rmse_diff": 0,
        "bias": 0,
        "sd": None,
        "coverage": None,
        "overlap": None, 
        "residual": None
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


def plot_dml_rmse_vs_residual_variance(
    df: pd.DataFrame,
    tau_hat_col: str = "dml_tau_hat",
    tau_true_col: str = "tau_true",
    resid_var_col: str = "resid_var_d",
    alpha_y_col: str = "alpha_y",
    alpha_d_col: str = "alpha_d",
    kappa_col: str = "kappa",
    average_over_alpha_d: bool = True,
    title: str = "DML RMSE vs Residualized Treatment Variance",
):
    """
    Plot DML RMSE against residualized treatment variance.

    RMSE is computed within each group as:
        sqrt(mean((tau_hat - tau_true)^2))

    Recommended interpretation:
    - x-axis: Var(D - e_hat(X))
    - y-axis: DML RMSE
    - separate lines by alpha_y
    - optionally average over alpha_d to keep the figure clean
    """

    required = {
        tau_hat_col,
        tau_true_col,
        resid_var_col,
        alpha_y_col,
        alpha_d_col,
        kappa_col,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_plot = df.copy()
    df_plot["dml_sq_error"] = (df_plot[tau_hat_col] - df_plot[tau_true_col]) ** 2

    group_cols = [alpha_y_col, kappa_col]
    if not average_over_alpha_d:
        group_cols.append(alpha_d_col)

    plot_df = (
        df_plot.groupby(group_cols, as_index=False)
        .agg(
            dml_rmse=("dml_sq_error", lambda x: np.sqrt(np.mean(x))),
            resid_var_d=(resid_var_col, "mean"),
        )
        .sort_values(
            [alpha_y_col, kappa_col] +
            ([alpha_d_col] if not average_over_alpha_d else [])
        )
    )

    if average_over_alpha_d:
        plot_df["line_group"] = plot_df[alpha_y_col].astype(str)
        color_col = alpha_y_col
        symbol_col = None
    else:
        plot_df["line_group"] = (
            plot_df[alpha_y_col].astype(str) + "_" +
            plot_df[alpha_d_col].astype(str)
        )
        color_col = alpha_y_col
        symbol_col = alpha_d_col

    fig = px.line(
        plot_df,
        x="resid_var_d",
        y="dml_rmse",
        color=color_col,
        symbol=symbol_col,
        markers=True,
        line_group="line_group",
        hover_data=[kappa_col] + ([alpha_d_col] if not average_over_alpha_d else []),
        title=title,
        labels={
            "resid_var_d": "Residualised Treatment Variance",
            "dml_rmse": "DML RMSE",
            alpha_y_col: "Outcome nonlinearity (alpha_y)",
            alpha_d_col: "Treatment complexity (alpha_d)",
            kappa_col: "Overlap regime (kappa)",
        },
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="alpha_y",
    )
    #fig.update_xaxes(autorange="reversed")

    return fig


def _collapse_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Collapse a dataframe to the requested grouping level.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_cols : list[str]
        Columns to group by.
    value_col : str
        Metric column to aggregate.
    agg : str
        Aggregation method, e.g. 'mean', 'median'.

    Returns
    -------
    pd.DataFrame
        Grouped dataframe with one row per group.
    """
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")

    grouped = (
        df[group_cols + [value_col]]
        .groupby(group_cols, as_index=False)
        .agg(**{value_col: (value_col, agg)})
    )
    return grouped


def plot_residual_variance_vs_kappa(
    df: pd.DataFrame,
    resid_var_col: str = "resid_var_d",
    kappa_col: str = "kappa",
    alpha_d_col: str = "alpha_d",
    alpha_y_col: str = "alpha_y",
    agg: str = "mean",
    average_over_alpha_y: bool = True,
    title: str = "Residualized Treatment Variance vs Overlap Regime",
):
    """
    Plot Var(D - e_hat(X)) against kappa.

    Recommended interpretation:
    - x-axis: kappa
    - y-axis: Var(D - e_hat(X))
    - separate lines by alpha_d
    - optionally average over alpha_y
    """
    required = {resid_var_col, kappa_col, alpha_d_col, alpha_y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if average_over_alpha_y:
        plot_df = _collapse_metric(
            df=df,
            group_cols=[kappa_col, alpha_d_col],
            value_col=resid_var_col,
            agg=agg,
        )
    else:
        plot_df = _collapse_metric(
            df=df,
            group_cols=[kappa_col, alpha_d_col, alpha_y_col],
            value_col=resid_var_col,
            agg=agg,
        )

    fig = px.line(
        plot_df,
        x=kappa_col,
        y=resid_var_col,
        color=alpha_d_col,
        markers=True,
        line_group=alpha_d_col,
        title=title,
        labels={
            kappa_col: "Overlap regime (kappa)",
            resid_var_col: "Var(D - ê(X))",
            alpha_d_col: "Treatment complexity (alpha_d)",
            alpha_y_col: "Outcome nonlinearity (alpha_y)",
        },
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="alpha_d",
        xaxis=dict(tickmode="linear"),
    )

    if not average_over_alpha_y:
        fig = px.line(
            plot_df,
            x=kappa_col,
            y=resid_var_col,
            color=alpha_d_col,
            facet_col=alpha_y_col,
            facet_col_wrap=3,
            markers=True,
            title=title,
            labels={
                kappa_col: "Overlap regime (kappa)",
                resid_var_col: "Var(D - ê(X))",
                alpha_d_col: "Treatment complexity (alpha_d)",
                alpha_y_col: "Outcome nonlinearity (alpha_y)",
            },
        )
        fig.update_traces(line=dict(width=2))
        fig.update_layout(
            template="plotly_white",
            legend_title_text="alpha_d",
            xaxis=dict(tickmode="linear"),
        )

    return fig


def plot_resid_var_vs_alpha_d(
    df: pd.DataFrame,
    resid_var_col: str = "estimated_resid_var",
    alpha_d_col: str = "alpha_d",
    kappa_col: str = "kappa",
    alpha_y_col: str = "alpha_y",
    agg: str = "mean",
    average_over_alpha_y: bool = True,
    title: str = "Residualized Treatment Variance vs Treatment Complexity",
):
    """
    Plot residualized treatment variance against alpha_d.

    Recommended:
    - x-axis: alpha_d
    - y-axis: Var(D - e_hat(X))
    - lines by kappa
    - average over alpha_y (since outcome complexity shouldn't matter here)
    """

    required = {resid_var_col, alpha_d_col, kappa_col, alpha_y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Collapse data
    if average_over_alpha_y:
        plot_df = (
            df.groupby([alpha_d_col, kappa_col], as_index=False)
            .agg({resid_var_col: agg})
        )
    else:
        plot_df = (
            df.groupby([alpha_d_col, kappa_col, alpha_y_col], as_index=False)
            .agg({resid_var_col: agg})
        )

    fig = px.line(
        plot_df,
        x=alpha_d_col,
        y=resid_var_col,
        color=kappa_col,
        markers=True,
        title=title,
        labels={
            alpha_d_col: "Treatment complexity (alpha_d)",
            resid_var_col: "Residualized treatment variance",
            kappa_col: "Overlap regime (kappa)",
        },
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="kappa",
    )

    return fig

def _check_required_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def plot_m_error_vs_alpha_y(
    df: pd.DataFrame,
    m_mse_col: str = "m_mse",
    alpha_y_col: str = "alpha_y",
    kappa_col: str = "kappa",
    alpha_d_col: str = "alpha_d",
    average_over_alpha_d: bool = True,
    agg: str = "mean",
    title: str = "Outcome Nuisance Error vs Outcome Nonlinearity",
):
    """
    Plot outcome nuisance error against alpha_y.
    Lines are split by kappa.
    """
    required = {m_mse_col, alpha_y_col, kappa_col, alpha_d_col}
    _check_required_columns(df, required)

    group_cols = [alpha_y_col, kappa_col]
    if not average_over_alpha_d:
        group_cols.append(alpha_d_col)

    plot_df = (
        df.groupby(group_cols, as_index=False)
        .agg({m_mse_col: agg})
        .sort_values(group_cols)
    )

    fig = px.line(
        plot_df,
        x=alpha_y_col,
        y=m_mse_col,
        color=kappa_col,
        markers=True,
        title=title,
        labels={
            alpha_y_col: "Outcome nonlinearity (alpha_y)",
            m_mse_col: "Outcome nuisance MSE",
            kappa_col: "Overlap regime (kappa)",
        },
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="kappa",
    )
    return fig


def plot_e_error_vs_alpha_d(
    df: pd.DataFrame,
    e_mse_col: str = "e_mse",
    alpha_d_col: str = "alpha_d",
    kappa_col: str = "kappa",
    alpha_y_col: str = "alpha_y",
    average_over_alpha_y: bool = True,
    agg: str = "mean",
    title: str = "Propensity Nuisance Error vs Treatment Complexity",
):
    """
    Plot propensity nuisance error against alpha_d.
    Lines are split by kappa.
    """
    required = {e_mse_col, alpha_d_col, kappa_col, alpha_y_col}
    _check_required_columns(df, required)

    group_cols = [alpha_d_col, kappa_col]
    if not average_over_alpha_y:
        group_cols.append(alpha_y_col)

    plot_df = (
        df.groupby(group_cols, as_index=False)
        .agg({e_mse_col: agg})
        .sort_values(group_cols)
    )

    fig = px.line(
        plot_df,
        x=alpha_d_col,
        y=e_mse_col,
        color=kappa_col,
        markers=True,
        title=title,
        labels={
            alpha_d_col: "Treatment complexity (alpha_d)",
            e_mse_col: "Propensity nuisance MSE",
            kappa_col: "Overlap regime (kappa)",
        },
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="kappa",
    )
    return fig



def plot_resid_var_vs_alpha_d(
    df: pd.DataFrame,
    resid_var_col: str = "estimated_resid_var",
    alpha_d_col: str = "alpha_d",
    kappa_col: str = "kappa",
    alpha_y_col: str = "alpha_y",
    agg: str = "mean",
    average_over_alpha_y: bool = True,
    title: str = "Residualized Treatment Variance vs Treatment Complexity",
):
    """
    Plot residualized treatment variance against alpha_d.

    Recommended:
    - x-axis: alpha_d
    - y-axis: Var(D - e_hat(X))
    - lines by kappa
    - average over alpha_y (since outcome complexity shouldn't matter here)
    """

    required = {resid_var_col, alpha_d_col, kappa_col, alpha_y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Collapse data
    if average_over_alpha_y:
        plot_df = (
            df.groupby([alpha_d_col, kappa_col], as_index=False)
            .agg({resid_var_col: agg})
        )
    else:
        plot_df = (
            df.groupby([alpha_d_col, kappa_col, alpha_y_col], as_index=False)
            .agg({resid_var_col: agg})
        )

    fig = px.line(
        plot_df,
        x=alpha_d_col,
        y=resid_var_col,
        color=kappa_col,
        markers=True,
        title=title,
        labels={
            alpha_d_col: "Treatment complexity (alpha_d)",
            resid_var_col: "Residualized treatment variance",
            kappa_col: "Overlap regime (kappa)",
        },
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        template="plotly_white",
        legend_title_text="kappa",
    )

    return fig

def plot_interaction(df):
    import numpy as np
    import plotly.express as px

    df_plot = df.copy()
    df_plot["tau_sq_error"] = (df_plot["dml_tau_hat"] - df_plot["tau_true"]) ** 2

    plot_df = (
        df_plot.groupby(["kappa", "alpha_y", "alpha_d"], as_index=False)
        .agg(
            tau_rmse=("tau_sq_error", lambda x: np.sqrt(np.mean(x))),
            m_mse=("m_mse", "mean"),
            resid_var_d=("estimated_resid_var", "mean"),
        )
    )

    plot_df["interaction"] = plot_df["m_mse"] / plot_df["resid_var_d"]

    fig = px.scatter(
        plot_df,
        x="interaction",
        y="tau_rmse",
        color="kappa",
        trendline="ols",
        labels={
            "interaction": "Outcome error / residual variation",
            "tau_rmse": "DML RMSE",
            "kappa": "Overlap (κ)"
        },
        title="DML RMSE as a Function of Outcome Error and Residual Variation", 
        trendline_color_override="black"
    )

    fig.update_traces(marker=dict(size=10, opacity=0.85))

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
    )

    return fig
