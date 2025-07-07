import streamlit as st
import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import base64


st.set_page_config(layout="wide")


def set_session_state():
    # set default values
    if "df" not in st.session_state:
        st.session_state.df = None
    if "nome_data" not in st.session_state:
        st.session_state.nome_data = "DATA"
    if "nome_qty" not in st.session_state:
        st.session_state.nome_qty = "QTY"
    if "is_data" not in st.session_state:
        st.session_state.is_data = True


def main():
    # Para facilitar o debug
    # df = pd.read_csv("AirPassengers.csv")
    # df["DATA"] = pd.to_datetime(df["DATA"])
    set_session_state()

    st.title("Bem vindo a ferramenta de séries temporais SARIMAX")
    st.markdown(
        """
    A sigla SARIMAX significa **S**easonal **A**uto**R**egressive **I**ntegrated **M**oving-**A**verage with e**X**ogenous regressors, onde cada termo representa:

    - **S** - Para sazonalidade, por exemplo, todo mês de dezembro ocorre um crescimento das vendas devido ao natal;
    - **AR** - Utiliza os termos passados para explicar os futuros, os parâmetros são indicados pela letra grega $\\phi{}$;
    - **I** - Diferencia a série para atender aos requisitos de estacionariedade;
    - **MA** - Utiliza os erros passados para explicar os valores futuros, os parâmetros são indicados pela letra grega $\\theta{}$;
    - **X** - Utiliza uma variável exógena para explicar a série. Exemplo, utilizar o risco país para explicar a cotação do dólar.

    Inicialmente temos que importar um dataset para realização da modelagem:  
    - O dataset pode ter o formato csv, xls ou xlsx;
    - O nome do campo de data e do valor para referência devem ser especificados abaixo;
    - Você terá a oportunidade de escolher os melhores parâmetros para o modelo posteriormente.
    """
    )

    c_nome_data, c_nome_qty, c_is_data = st.columns(3)

    st.session_state.nome_data = c_nome_data.text_input(
        "Nome do campo de data:",
        value=st.session_state.nome_data,
        max_chars=256,
        key=None,
        type="default",
    )
    st.session_state.nome_qty = c_nome_qty.text_input(
        "Nome do campo de quantidade de eventos:",
        value=st.session_state.nome_qty,
        max_chars=256,
        key=None,
        type="default",
    )
    st.session_state.is_data = c_is_data.checkbox(
        "O index é uma data?", value=st.session_state.is_data, key=None
    )

    st.write(
        """
    Seleção de Arquivo (arquivos de exemplo: [Icecream](https://www3.nd.edu/~busiforc/handouts/Data%20and%20Stories/regression/ice%20cream%20consumption/icecream.html) e
    [AirPassengers](https://www.kaggle.com/datasets/chirag19/air-passengers)):
    """
    )

    file_zerar, file_Icecream, file_AirPassengers = st.columns(3)

    if file_zerar.button("Cancelar seleção do arquivo exemplo", key="zerar"):
        uploaded_file = None
        st.session_state.df = None

    if file_Icecream.button("Arquivo exemplo Icecream", key="Icecream"):
        st.session_state.df = pd.read_csv("datasets/Icecream.csv")
        st.session_state.is_data = False
        st.session_state.nome_data = "DATA"
        st.session_state.nome_qty = "QTY"

    if file_AirPassengers.button("Arquivo exemplo AirPassengers", key="AirPassengers"):
        st.session_state.df = pd.read_csv("datasets/AirPassengers.csv")
        st.session_state.is_data = True
        st.session_state.nome_data = "DATA"
        st.session_state.nome_qty = "QTY"

    uploaded_file = st.file_uploader(
        "Selecione o arquivo ou arraste o arquivo:",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        if st.session_state.nome_data == "":
            st.write("Por favor, preencha o nome do campo de data.")
        elif st.session_state.nome_qty == "":
            st.write("Por favor, preencha o nome do campo de quantidade.")
        else:
            st.write(
                'Importação do arquivo "'
                + uploaded_file.name
                + '" foi realizada com sucesso'
            )

            if uploaded_file.name.split(".")[-1].upper() == "CSV":
                st.session_state.df = pd.read_csv(uploaded_file, sep=None)
            elif uploaded_file.name.split(".")[-1].upper() in ("XLSX", "XLS"):
                st.session_state.df = pd.read_excel(uploaded_file)
            if st.session_state.is_data:
                st.session_state.df[st.session_state.nome_data] = pd.to_datetime(
                    st.session_state.df[st.session_state.nome_data]
                )

    if st.session_state.df is not None:
        st.subheader("Prévia dos dados importados")
        st.dataframe(st.session_state.df.head(), use_container_width=True)

        fig = plot_grafico_1(
            st.session_state.df,
            st.session_state.nome_data,
            st.session_state.nome_qty,
            st.session_state.is_data,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("***")

        st.header("Modelo SARIMAX:")

        st.write(
            "A técnica ARIMA (AutoRegressive Integrated Moving Average) utiliza os pontos \
            anteriores de uma série temporal para realizar a predição dos pontos seguintes."
        )

        y = st.session_state.df.set_index(st.session_state.nome_data)
        tests_adf_autocorr(
            y[st.session_state.nome_qty],
            "Testes estatísticos para a série sem diferenciação",
        )

        ##################################################
        # Parâmetros SARIMA
        ##################################################
        par_ARIMA = dict()
        st.markdown(
            """
        **Escolha os parâmetros do modelo:**  
        **Não sazonais:**
        """
        )
        c_par_const_, c_par_AR_, c_par_I_, c_par_MA_ = st.columns(4)
        c_par_AR, c_par_I, c_par_MA, c_par_const = (
            c_par_AR_.empty(),
            c_par_I_.empty(),
            c_par_MA_.empty(),
            c_par_const_.empty(),
        )
        par_ARIMA["p"] = c_par_AR.number_input(
            "p:", min_value=0, max_value=1000, value=1, step=1, format="%d", key="p"
        )
        par_ARIMA["d"] = c_par_I.number_input(
            "d:", min_value=0, max_value=1000, value=0, step=1, format="%d", key="d"
        )
        par_ARIMA["q"] = c_par_MA.number_input(
            "q:", min_value=0, max_value=1000, value=0, step=1, format="%d", key="q"
        )
        if c_par_const.checkbox("Modelo tem constante?", value=False, key="trend"):
            par_ARIMA["trend"] = "c"
        else:
            par_ARIMA["trend"] = "n"

        st.markdown(
            """
        **Sazonais:**
        """
        )
        sazo_c_par_n_, sazo_c_par_AR_, sazo_c_par_I_, sazo_c_par_MA_ = st.columns(4)
        sazo_c_par_AR, sazo_c_par_I, sazo_c_par_MA, sazo_c_par_n = (
            sazo_c_par_AR_.empty(),
            sazo_c_par_I_.empty(),
            sazo_c_par_MA_.empty(),
            sazo_c_par_n_.empty(),
        )
        par_ARIMA["P"] = sazo_c_par_AR.number_input(
            "P (Sazonal):",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
            format="%d",
            key="P",
        )
        par_ARIMA["D"] = sazo_c_par_I.number_input(
            "D (Sazonal):",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
            format="%d",
            key="D",
        )
        par_ARIMA["Q"] = sazo_c_par_MA.number_input(
            "Q (Sazonal):",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
            format="%d",
            key="Q",
        )
        par_ARIMA["n"] = sazo_c_par_n.number_input(
            "Período da sazonalidade (n):",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
            format="%d",
            key="n",
        )

        if par_ARIMA["n"] == 1:
            st.markdown("A sazonalidade precisa ser maior do que 1 ou nula")
        elif par_ARIMA["n"] == 0 and (
            par_ARIMA["P"] > 0 or par_ARIMA["D"] > 0 or par_ARIMA["Q"] > 0
        ):
            st.markdown(
                "Para utilização dos argumentos sazonais (P, D, Q), a sazonalidade deve ser maior do que 1."
            )
        elif par_ARIMA["trend"] == "c" and (par_ARIMA["d"] > 0 or par_ARIMA["D"] > 0):
            st.markdown("O modelo com constante não pode ser diferenciado.")
        else:
            param_models = st.empty()
            param_models_text(param_models, par_ARIMA)

            ##################################################
            # Variáveis exógenas
            ##################################################

            exogenas = y.drop(st.session_state.nome_qty, axis=1).columns

            if len(exogenas) > 0:
                st.markdown(
                    """
                **Exógenas (X):**
                """
                )
                selected_exogens = st.multiselect(
                    "Selecione as exógenas a serem utilizadas no modelo:", exogenas
                )
            else:
                selected_exogens = []

            ##################################################
            # Separação entre treinamento e teste
            ##################################################

            test_size_input = st.number_input(
                "Tamanho do conjunto de treinamento (Pode ser 0, se menor que 1, será considerado um percentual), o padrão é utilizar 20% do dataset:",
                min_value=0.0,
                max_value=10000.0,
                value=np.floor(y.shape[0] * 0.2),
                step=0.01,
            )
            if test_size_input > 0 and test_size_input < 1:
                train_size = int(y.shape[0] * (1 - test_size_input))
                y_train = y[:train_size]
                y_test = y[train_size:]
            elif test_size_input >= 1:
                y_train = y[: y.shape[0] - int(test_size_input)]
                y_test = y[y.shape[0] - int(test_size_input) :]
            else:
                y_train = y
                y_test = None

            ##################################################
            # auto.arima
            ##################################################

            if len(selected_exogens) == 0:
                st.markdown(
                    """Você pode utilizar um método automático para escolher os parâmetros do modelo. 
                Utilizamos a biblioteca [pmdarima](https://alkaline-ml.com/pmdarima/), que é baseada na função auto.arima 
                do pacote forecast da linguagem R.  
                Se a série for sazonal você deve configurar acima a sazonalidade e depois executar o método de seleção."""
                )

                if st.button("Estimar com auto.arima"):
                    if par_ARIMA["n"] > 0:
                        model_auto_arima = pm.auto_arima(
                            y_train[st.session_state.nome_qty],
                            seasonal=True,
                            m=par_ARIMA["n"],
                            with_intercept=False,
                            trend="n",
                        )
                    else:
                        model_auto_arima = pm.auto_arima(
                            y_train[st.session_state.nome_qty],
                            seasonal=False,
                            with_intercept=False,
                            trend="n",
                        )
                    par_ARIMA["p"] = model_auto_arima.get_params()["order"][0]
                    par_ARIMA["d"] = model_auto_arima.get_params()["order"][1]
                    par_ARIMA["q"] = model_auto_arima.get_params()["order"][2]

                    par_ARIMA["P"] = model_auto_arima.get_params()["seasonal_order"][0]
                    par_ARIMA["D"] = model_auto_arima.get_params()["seasonal_order"][1]
                    par_ARIMA["Q"] = model_auto_arima.get_params()["seasonal_order"][2]
                    par_ARIMA["n"] = model_auto_arima.get_params()["seasonal_order"][3]

                    c_par_AR.number_input(
                        "p:",
                        min_value=0,
                        max_value=1000,
                        value=par_ARIMA["p"],
                        step=1,
                        format="%d",
                        key="pn",
                    )
                    c_par_I.number_input(
                        "d:",
                        min_value=0,
                        max_value=1000,
                        value=par_ARIMA["d"],
                        step=1,
                        format="%d",
                        key="dn",
                    )
                    c_par_MA.number_input(
                        "q:",
                        min_value=0,
                        max_value=1000,
                        value=par_ARIMA["q"],
                        step=1,
                        format="%d",
                        key="qn",
                    )

                    sazo_c_par_AR.number_input(
                        "P (Sazonal):",
                        min_value=0,
                        max_value=1000,
                        value=par_ARIMA["P"],
                        step=1,
                        format="%d",
                        key="Pn",
                    )
                    sazo_c_par_I.number_input(
                        "D (Sazonal):",
                        min_value=0,
                        max_value=1000,
                        value=par_ARIMA["D"],
                        step=1,
                        format="%d",
                        key="Dn",
                    )
                    sazo_c_par_MA.number_input(
                        "Q (Sazonal):",
                        min_value=0,
                        max_value=1000,
                        value=par_ARIMA["Q"],
                        step=1,
                        format="%d",
                        key="Qn",
                    )
                    sazo_c_par_n.number_input(
                        "Período da sazonalidade (n):",
                        min_value=0,
                        max_value=1000,
                        value=par_ARIMA["n"],
                        step=1,
                        format="%d",
                        key="nn",
                    )

                    param_models_text(param_models, par_ARIMA)

                    st.text(
                        "Parâmetros estimados com auto.arima. Foi considerado uma sazonalidade {}".format(
                            par_ARIMA["n"]
                        )
                    )
            else:
                st.markdown(
                    "Não é possível utilizar o autorima quando são utilizadas variáveis exógenas"
                )

            ##################################################
            # Refaz os testes ADF e ACF/PACF caso exista diferenciação
            ##################################################
            if par_ARIMA["d"] > 0:
                tests_adf_autocorr(
                    y_train[st.session_state.nome_qty].diff(periods=par_ARIMA["d"])[
                        par_ARIMA["d"] :
                    ],
                    "Testes estatísticos para a série com d = {}".format(
                        par_ARIMA["d"]
                    ),
                )

            ##################################################
            # Treina modelo
            ##################################################
            if len(selected_exogens) == 0:
                model = ARIMA(
                    y_train[st.session_state.nome_qty],
                    order=(par_ARIMA["p"], par_ARIMA["d"], par_ARIMA["q"]),
                    seasonal_order=(
                        par_ARIMA["P"],
                        par_ARIMA["D"],
                        par_ARIMA["Q"],
                        par_ARIMA["n"],
                    ),
                    trend=par_ARIMA["trend"],
                )
            else:
                y_train_exogs = y_train.drop(st.session_state.nome_qty, axis=1)[
                    selected_exogens
                ]
                y_test_exogs = y_test.drop(st.session_state.nome_qty, axis=1)[
                    selected_exogens
                ]
                model = ARIMA(
                    y_train[st.session_state.nome_qty],
                    exog=y_train_exogs,
                    order=(par_ARIMA["p"], par_ARIMA["d"], par_ARIMA["q"]),
                    seasonal_order=(
                        par_ARIMA["P"],
                        par_ARIMA["D"],
                        par_ARIMA["Q"],
                        par_ARIMA["n"],
                    ),
                    trend=par_ARIMA["trend"],
                )
            model_fit = model.fit()

            model_summary = st.expander("Veja os parâmetros do modelo:")
            with model_summary:
                st.write(model_fit.summary())

            ##################################################
            # Define a quantidade de pontos da projeção e projeta
            ##################################################

            if len(selected_exogens) == 0:
                if y_test is not None:
                    tamanho_projecao = st.number_input(
                        "Define a quantidade de pontos a serem projetados (por padrão 50% do tamanho do dataset):",
                        min_value=y_test.shape[0],
                        max_value=10000,
                        value=int(y.shape[0] * 0.5),
                        step=1,
                    )
                else:
                    tamanho_projecao = st.number_input(
                        "Define a quantidade de pontos a serem projetados (por padrão 50% do tamanho do dataset):",
                        min_value=0,
                        max_value=10000,
                        value=int(y.shape[0] * 0.5),
                        step=1,
                    )

                y_pred = model_fit.forecast(steps=tamanho_projecao)
            else:
                y_pred = model_fit.forecast(
                    steps=y_test_exogs.shape[0], exog=y_test_exogs
                )
                y_pred.index = y_pred.index + 1

            ##################################################
            # Plota Forecast
            ##################################################
            fig_forecast = plot_grafico_projecao(
                y_train[st.session_state.nome_qty],
                y_test[st.session_state.nome_qty],
                y_pred,
                st.session_state.nome_data,
                st.session_state.nome_qty,
                st.session_state.is_data,
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            ##################################################
            # Métricas
            ##################################################

            if y_test is not None:
                metricas_analise = st.expander("Métricas:")
                with metricas_analise:
                    y_pred_metrics = y_pred[: y_test.shape[0]]
                    metric_MAPE = mean_absolute_percentage_error(
                        y_test[st.session_state.nome_qty], y_pred_metrics
                    )
                    metric_RMSE = np.sqrt(
                        mean_absolute_error(
                            y_test[st.session_state.nome_qty], y_pred_metrics
                        )
                    )
                    st.markdown(
                        f"""
                    **Métricas:**  
                    
                    | Métrica  | Valor |
                    |----------|--------|
                    | MAPE     | {metric_MAPE:.5f} |
                    | RMSE     | {metric_RMSE:.5f} |  
                    """
                    )

            ##################################################
            # Análise dos resíduos
            ##################################################
            resid_analise = st.expander("Análise dos resíduos:")
            with resid_analise:
                residuos = model_fit.resid[1:]
                st.markdown("**Funções de autocorrelação dos resíduos:**")
                c_acf_resi, c_pacf_resi = st.columns(2)
                acf_resi = ts.acf(
                    residuos, nlags=min(int(residuos.shape[0] / 2 - 1), 25), alpha=0.05
                )
                c_acf_resi.plotly_chart(
                    plot_auto_correlation(acf_resi, 25, "ACF dos resíduos"),
                    use_container_width=True,
                )
                pacf_resi = ts.pacf(
                    residuos, nlags=min(int(residuos.shape[0] / 2 - 1), 25), alpha=0.05
                )
                c_pacf_resi.plotly_chart(
                    plot_auto_correlation(pacf_resi, 25, "PACF dos resíduos"),
                    use_container_width=True,
                )

                lb_results = sm.stats.diagnostic.acorr_ljungbox(
                    model_fit.resid.values, lags=1, return_df=True
                )["lb_pvalue"].values[0]
                if lb_results > 0.05:
                    text_result_lb = f"- **Não existe correlação serial do lag 1** 👍. P-Valor: {lb_results:.3f}"
                else:
                    text_result_lb = f"- **Existe correlação serial do lag 1** 👎. P-Valor: {lb_results:.3f}"

                st.markdown(
                    """
                **Teste de Ljung–Box para autocorrelação.** 
                - **Hipóteses:**
                    - Hipótese nula ($H_0$): Não existe correlação serial de ordem até "p" (que é escolhido por quem está utilizando o teste);  
                    - Hipótese alternativa ($H_1$): Existe correlação serial de ao menos uma ordem até "p".  
                """
                    + text_result_lb
                )

            ##################################################
            # Download projeção
            ##################################################
            if y_test is not None:
                df_projecao = (
                    pd.concat(
                        [
                            y_train[st.session_state.nome_qty],
                            y_test[st.session_state.nome_qty],
                            y_pred,
                        ],
                        axis=1,
                    )
                    .fillna(0)
                    .reset_index()
                )
                df_projecao.columns = ["Data", "Treinamento", "Teste", "Projeção"]
            else:
                df_projecao = (
                    pd.concat(
                        [
                            y_train[st.session_state.nome_qty],
                            y_pred[st.session_state.nome_qty],
                        ],
                        axis=1,
                    )
                    .fillna(0)
                    .reset_index()
                )
                df_projecao.columns = ["Data", "Treinamento", "Projeção"]
            st.markdown("Faça o download da projeção:  ")
            download_dataframe(df_projecao)


def download_dataframe(df, file_name="captura.csv"):
    """Renderiza um botão para download do dataframe como CSV."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=file_name,
        mime="text/csv",
    )


def tests_adf_autocorr(y, texto):
    statistical_tests = st.expander(texto)
    with statistical_tests:
        # Teste de estacionariedade
        st.markdown("***")
        st.markdown("**Teste de estacionariedade de Dickey-Fuller Aumentado (ADF):**")
        check_adfuller(y)

        # Plota ACF e PACF
        st.markdown("***")
        c_acf, c_pacf = st.columns(2)
        acf = ts.acf(y, nlags=min(int(y.shape[0] / 2 - 1), 25), alpha=0.05)
        c_acf.plotly_chart(
            plot_auto_correlation(acf, 25, "ACF"), use_container_width=True
        )
        pacf = ts.pacf(y, nlags=min(int(y.shape[0] / 2 - 1), 25), alpha=0.05)
        c_pacf.plotly_chart(
            plot_auto_correlation(pacf, 25, "PACF"), use_container_width=True
        )


def param_models_text(param_models, par_ARIMA):
    param_models.markdown(
        f"""
    Modelo sendo utilizado (notação $(p, d, q)x(P, D, Q, s)$):
    $({par_ARIMA["p"]}, {par_ARIMA["d"]}, {par_ARIMA["q"]})x({par_ARIMA["P"]}, {par_ARIMA["D"]}, {par_ARIMA["Q"]}, {par_ARIMA["n"]})$
    """
    )


def check_adfuller(y):
    est = ts.adfuller(y)[1]
    if est <= 0.05:
        st.markdown("👍 A série **é estacionária** com p-valor de : {:0.4f}".format(est))
    else:
        st.markdown(
            "👎 A série **não é estacionária** com p-valor de : {:0.4f}".format(est)
        )


@st.cache_data
def plot_auto_correlation(serie, n, versao):
    x = list(range(0, n + 1))
    y = serie[0]
    y_upper = serie[1][:, 1] - y
    y_lower = serie[1][:, 0] - y

    fig = go.Figure(
        [
            go.Scatter(x=x, y=y, line_color="red", mode="markers"),
            go.Bar(x=x, y=y, width=0.1),
            go.Scatter(
                x=x, y=y_lower, fill="tonexty", mode="none", line_color="indigo"
            ),
            go.Scatter(
                x=x, y=y_upper, fill="tonexty", mode="none", line_color="indigo"
            ),
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
    # Based in: https://plotly.com/python/range-slider/

    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(df[nome_data]), y=list(df[nome_qty])))

    # Set title
    fig.update_layout(
        title_text="Gráfico da série importada",
        width=1200,
        height=400,
    )

    # Add range slider
    if is_data:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            dict(
                                count=6, label="6m", step="month", stepmode="backward"
                            ),
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


def plot_grafico_projecao(X_train, X_test, X_pred, nome_data, nome_qty, is_data=True):
    fig = go.Figure()
    X_train = X_train.reset_index()
    X_pred = X_pred.reset_index()
    fig.add_trace(
        go.Scatter(
            x=list(X_train[nome_data]),
            y=list(X_train[nome_qty]),
            name="Conjunto Treinamento",
        )
    )

    if X_test is not None:
        X_test = X_test.reset_index()
        fig.add_trace(
            go.Scatter(
                x=list(X_test[nome_data]),
                y=list(X_test[nome_qty]),
                name="Conjunto Teste",
            )
        )

    fig.add_trace(
        go.Scatter(x=X_pred["index"], y=X_pred["predicted_mean"], name="Projeção")
    )

    fig.update_layout(
        title_text="Gráfico Observações e Projeção", width=1200, height=400
    )

    if is_data:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            dict(
                                count=6, label="6m", step="month", stepmode="backward"
                            ),
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


if __name__ == "__main__":
    main()
