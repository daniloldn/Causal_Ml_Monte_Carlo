import plotly.express as px
import pandas as pd

from src.metrics import rmse_diff, bias, sd, coverage



def rmse_frontier(df: pd.Dataframe):

    rmse = rmse_diff(df)

    fig = px.density_heatmap(
    rmse,
    x="alpha_d",
    y="alpha_y",
    z="rmse_diff",
    histfunc="avg",
    color_continuous_scale="RdBu",
    title="RMSE Difference: OLS - DML"
    )

    fig.update_layout(
    xaxis_title="alpha_d",
    yaxis_title="alpha_y",
    coloraxis_colorbar_title="RMSE diff"
    )

    fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=rmse.columns,
        ticktext=[str(x) for x in rmse.columns]
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=rmse.index,
        ticktext=[str(y) for y in rmse.index]
    )
    )

    fig.show()

    return None


def bias_ols(df: pd.DataFrame) -> pd.DataFrame:

    bias_ols = bias(df)

    fig = px.density_heatmap(bias_ols,
                         y= "alpha_y",
                         x = "alpha_d", 
                         z = "OLS")


    fig.update_layout(
    xaxis_title="alpha_d",
    yaxis_title="alpha_y",
    coloraxis_colorbar_title="Bias OLS"
    )

    fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=bias_ols.columns,
        ticktext=[str(x) for x in bias_ols.columns]
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=bias_ols.index,
        ticktext=[str(y) for y in bias_ols.index]
    )
    )

    fig.show()

    return None

def bias_dml(df: pd.DataFrame) -> pd.DataFrame:

    bias_dml = bias(df)

    fig = px.density_heatmap(bias_dml,
                         y= "alpha_y",
                         x = "alpha_d", 
                         z = "DML")


    fig.update_layout(
    xaxis_title="alpha_d",
    yaxis_title="alpha_y",
    coloraxis_colorbar_title="Bias DML"
    )

    fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=bias_dml.columns,
        ticktext=[str(x) for x in bias_dml.columns]
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=bias_dml.index,
        ticktext=[str(y) for y in bias_dml.index]
    )
    )

    fig.show()

    return None


def sd_dml(df: pd.DataFrame) -> pd.DataFrame:

    sd_dml = sd(df)

    fig = px.density_heatmap(sd_dml,
                         y= "alpha_y",
                         x = "alpha_d", 
                         z = "DML")


    fig.update_layout(
    xaxis_title="alpha_d",
    yaxis_title="alpha_y",
    coloraxis_colorbar_title="SD DML"
    )

    fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=sd_dml.columns,
        ticktext=[str(x) for x in sd_dml.columns]
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=sd_dml.index,
        ticktext=[str(y) for y in sd_dml.index]
    )
    )

    fig.show()

    return None


def sd_ols(df: pd.DataFrame) -> pd.DataFrame:

    sd_ols = sd(df)

    fig = px.density_heatmap(sd_ols,
                         y= "alpha_y",
                         x = "alpha_d", 
                         z = "OLS")


    fig.update_layout(
    xaxis_title="alpha_d",
    yaxis_title="alpha_y",
    coloraxis_colorbar_title="SD OLS"
    )

    fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=sd_ols.columns,
        ticktext=[str(x) for x in sd_ols.columns]
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=sd_ols.index,
        ticktext=[str(y) for y in sd_ols.index]
    )
    )

    fig.show()

    return None

def coverage_dml(df: pd.DataFrame) -> pd.DataFrame:

    dml = coverage(df)

    fig = px.density_heatmap(dml,
                         y= "alpha_y",
                         x = "alpha_d", 
                         z = "DML")


    fig.update_layout(
    xaxis_title="alpha_d",
    yaxis_title="alpha_y",
    coloraxis_colorbar_title="Coverage DML"
    )

    fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=dml.columns,
        ticktext=[str(x) for x in dml.columns]
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=dml.index,
        ticktext=[str(y) for y in dml.index]
    )
    )

    fig.show()

    return None


def coverage_ols(df: pd.DataFrame) -> pd.DataFrame:

    ols = coverage(df)

    fig = px.density_heatmap(sd_ols,
                         y= "alpha_y",
                         x = "alpha_d", 
                         z = "OLS")


    fig.update_layout(
    xaxis_title="alpha_d",
    yaxis_title="alpha_y",
    coloraxis_colorbar_title="Coverage OLS"
    )

    fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=ols.columns,
        ticktext=[str(x) for x in ols.columns]
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=ols.index,
        ticktext=[str(y) for y in ols.index]
    )
    )

    fig.show()

    return None
