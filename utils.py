"""Utility functions for the SARIMAX Streamlit app."""
import streamlit as st
import statsmodels.tsa.stattools as ts
from pathlib import Path
import pandas as pd

from plots import plot_auto_correlation


def set_session_state():
    """Initialize default values in Streamlit session state."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "nome_data" not in st.session_state:
        st.session_state.nome_data = "DATA"
    if "nome_qty" not in st.session_state:
        st.session_state.nome_qty = "QTY"
    if "is_data" not in st.session_state:
        st.session_state.is_data = True


def download_dataframe(df, file_name="captura.csv"):
    """Render a button for downloading a dataframe as CSV."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=csv, file_name=file_name, mime="text/csv")


def tests_adf_autocorr(y, texto):
    """Display ADF test and autocorrelation plots."""
    statistical_tests = st.expander(texto)
    with statistical_tests:
        st.markdown("***")
        st.markdown("**Teste de estacionariedade de Dickey-Fuller Aumentado (ADF):**")
        check_adfuller(y)
        st.markdown("***")
        c_acf, c_pacf = st.columns(2)
        acf = ts.acf(y, nlags=min(int(y.shape[0] / 2 - 1), 25), alpha=0.05)
        c_acf.plotly_chart(plot_auto_correlation(acf, 25, "ACF"), use_container_width=True)
        pacf = ts.pacf(y, nlags=min(int(y.shape[0] / 2 - 1), 25), alpha=0.05)
        c_pacf.plotly_chart(plot_auto_correlation(pacf, 25, "PACF"), use_container_width=True)


def param_models_text(param_models, par_ARIMA):
    """Display the selected SARIMAX parameters."""
    param_models.markdown(
        f"""
    Modelo sendo utilizado (notação $(p, d, q)x(P, D, Q, s)$):
    $({par_ARIMA['p']}, {par_ARIMA['d']}, {par_ARIMA['q']})x({par_ARIMA['P']}, {par_ARIMA['D']}, {par_ARIMA['Q']}, {par_ARIMA['n']})$
    """
    )


def check_adfuller(y):
    """Run augmented Dickey-Fuller test and show the result."""
    est = ts.adfuller(y)[1]
    if est <= 0.05:
        st.markdown(f"👍 A série **é estacionária** com p-valor de : {est:.4f}")
    else:
        st.markdown(f"👎 A série **não é estacionária** com p-valor de : {est:.4f}")


def load_dataset(name: str) -> pd.DataFrame:
    """Return one of the bundled example datasets by name."""
    data_dir = Path(__file__).resolve().parent / "visual_sarimax" / "datasets"
    return pd.read_csv(data_dir / f"{name}.csv")
