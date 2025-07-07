# Plotting functions for the SARIMAX Streamlit app
import plotly.graph_objects as go
import streamlit as st
import statsmodels.tsa.stattools as ts

@st.cache_data
def plot_auto_correlation(serie, n, versao):
    """Create an autocorrelation or partial autocorrelation plot."""
    x = list(range(0, n + 1))
    y = serie[0]
    y_upper = serie[1][:, 1] - y
    y_lower = serie[1][:, 0] - y

    fig = go.Figure(
        [
            go.Scatter(x=x, y=y, line_color="red", mode="markers"),
            go.Bar(x=x, y=y, width=0.1),
            go.Scatter(x=x, y=y_lower, fill="tonexty", mode="none", line_color="indigo"),
            go.Scatter(x=x, y=y_upper, fill="tonexty", mode="none", line_color="indigo"),
        ]
    )
    fig.update_layout(
        showlegend=False,
        title=versao,
        xaxis_title="Lags",
        yaxis_title="Corr",
        width=500,
        height=400,
    )
    return fig

@st.cache_data
def plot_grafico_1(df, nome_data, nome_qty, is_data=True):
    """Plot the imported time series data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(df[nome_data]), y=list(df[nome_qty])))

    fig.update_layout(title_text="Gráfico da série importada", width=1200, height=400)

    if is_data:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=True),
                type="date",
            )
        )
    return fig

@st.cache_data
def plot_grafico_projecao(X_train, X_test, X_pred, nome_data, nome_qty, is_data=True):
    """Plot training, test and forecasted values."""
    fig = go.Figure()
    X_train = X_train.reset_index()
    X_pred = X_pred.reset_index()
    fig.add_trace(
        go.Scatter(x=list(X_train[nome_data]), y=list(X_train[nome_qty]), name="Conjunto Treinamento")
    )

    if X_test is not None:
        X_test = X_test.reset_index()
        fig.add_trace(
            go.Scatter(x=list(X_test[nome_data]), y=list(X_test[nome_qty]), name="Conjunto Teste")
        )

    fig.add_trace(go.Scatter(x=X_pred["index"], y=X_pred["predicted_mean"], name="Projeção"))

    fig.update_layout(title_text="Gráfico Observações e Projeção", width=1200, height=400)

    if is_data:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=True),
                type="date",
            )
        )
    return fig
