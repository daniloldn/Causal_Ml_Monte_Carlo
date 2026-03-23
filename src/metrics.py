import numpy as np 
import pandas as pd


def summarize_estimator(df: pd.DataFrame, tau_col: str, lower_col: str, upper_col: str) ->pd.Series:

    tau_true = df["tau_true"].iloc[0]
    tau_hat = df[tau_col]

    bias = (tau_hat - tau_true).mean()
    sd = tau_hat.std()
    rmse = np.sqrt(((tau_hat - tau_true) ** 2).mean())
    coverage = ((df[lower_col] <= tau_true) & (df[upper_col] >= tau_true)).mean()
    avg_ci_length = (df[upper_col] - df[lower_col]).mean()

    return pd.Series({
        "bias": bias,
        "sd": sd,
        "rmse": rmse,
        "coverage": coverage,
        "avg_ci_length": avg_ci_length
    })


def summarize_scenario(df: pd.DataFrame) -> pd.DataFrame:
    alpha_y = df["alpha_y"].iloc[0]
    alpha_d = df["alpha_d"].iloc[0]
    kappa = df["kappa"].iloc[0]

    ols_summary = summarize_estimator(
        df,
        tau_col="ols_tau_hat",
        lower_col="ols_ci_lower",
        upper_col="ols_ci_upper"
    )

    dml_summary = summarize_estimator(
        df,
        tau_col="dml_tau_hat",
        lower_col="dml_ci_lower",
        upper_col="dml_ci_upper"
    )

    out = pd.DataFrame([
        {"estimator": "OLS", **ols_summary.to_dict()},
        {"estimator": "DML", **dml_summary.to_dict()}
    ])

    out["alpha_y"] = alpha_y
    out["alpha_d"] = alpha_d
    out["kappa"] = kappa

    return out


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:

    
    scenario_summaries = []
    
    for (alpha_y, alpha_d, kappa), group in df.groupby(["alpha_y", "alpha_d", "kappa"]):
        summary = summarize_scenario(group)
        scenario_summaries.append(summary)
        summary_df = pd.concat(scenario_summaries, ignore_index=True)
        
    return summary_df


def metric_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Reshape one metric into wide form with OLS and DML as columns.
    """
    required = {"alpha_y", "alpha_d", "kappa", "estimator", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    wide = (
        df.pivot_table(
            index=["alpha_y", "alpha_d", "kappa"],
            columns="estimator",
            values=metric,
        )
        .reset_index()
    )

    wide.columns.name = None
    return wide


def rmse_diff_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a wide RMSE table and add RMSE difference = OLS - DML.
    Positive values mean DML performs better.
    """
    wide = metric_wide(df, "rmse")
    wide["rmse_diff"] = wide["OLS"] - wide["DML"]
    return wide


def bias_table(df: pd.DataFrame) -> pd.DataFrame:
    return metric_wide(df, "bias")


def sd_table(df: pd.DataFrame) -> pd.DataFrame:
    return metric_wide(df, "sd")


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    return metric_wide(df, "coverage")


def ci_length_table(df: pd.DataFrame) -> pd.DataFrame:
    return metric_wide(df, "avg_ci_length")