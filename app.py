import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

st.set_page_config(layout="wide")

def main():
    df = None
    st.title("Bem vindo a ferramenta de séries temporais ARIMA")
    st.markdown('''
    Inicialmente temos que importar um dataset para realização da modelagem:  
    - O dataset pode ter o formato csv, xls ou xlsx;
    - O nome do campo de data e do valor para referência devem ser especificados abaixo;
    - Você terá a oportunidade de escolher os melhores parâmetros para o modelo posteriormente.
    ''')

    nome_data = st.text_input("Nome do campo de data:", value='DATA', max_chars=256, key=None, type='default')
    nome_qty = st.text_input("Nome do campo de quantidade de eventos:", value='QTY', max_chars=256, key=None, type='default')

    uploaded_file = st.file_uploader("Selecione o arquivo ou arraste o arquivo:", type=['csv', 'xslx', 'xls'])

    if uploaded_file is not None:
        if nome_data == "":
            st.write("Por favor, preencha o nome do campo de data.")
        elif nome_qty == "":
            st.write("Por favor, preencha o nome do campo de quantidade.")
        else:
            st.write('Importação do arquivo "' + uploaded_file.name + '" foi realizada com sucesso')

            df = pd.read_csv(uploaded_file)
            df[nome_data] = pd.to_datetime(df[nome_data])
            # df.set_index(nome_data, inplace = True)

    if df is not None:
        fig = plot_grafico_1(df, nome_data, nome_qty)
        st.plotly_chart(fig)

        st.markdown('***')

        st.header("Modelo ARIMA:")

        st.write("A técnica ARIMA (AutoRegressive Integrated Moving Average) utiliza os pontos \
            anteriores de uma série temporal para realizar a predição dos pontos seguintes.")

        y = df[nome_qty]

        tests_adf_autocorr(y, 'Testes estatísticos para a série sem diferenciação')

        par_ARIMA = dict()

        st.markdown('''
        **Escolha os parâmetros do modelo:**  
        **Não sazonais:**
        ''')
        c_par_const_, c_par_AR_, c_par_I_, c_par_MA_ = st.beta_columns(4)
        c_par_AR, c_par_I, c_par_MA, c_par_const = c_par_AR_.empty(), c_par_I_.empty(), c_par_MA_.empty(), c_par_const_.empty()
        par_ARIMA["p"] = c_par_AR.number_input('p:', min_value=0, max_value=1000, value=1, step=1, format = "%d", key = "p")
        par_ARIMA["d"] = c_par_I.number_input('d:', min_value=0, max_value=1000, value=0, step=1, format = "%d", key = "d")
        par_ARIMA["q"] = c_par_MA.number_input('q:', min_value=0, max_value=1000, value=0, step=1, format = "%d", key = "q")
        if c_par_const.checkbox('Modelo tem constante?', value=False, key="trend"):
            par_ARIMA["trend"] = 'c'
        else:
            par_ARIMA["trend"] = 'n'


        st.markdown('''
        **Sazonais:**
        ''')
        sazo_c_par_n_, sazo_c_par_AR_, sazo_c_par_I_, sazo_c_par_MA_ = st.beta_columns(4)
        sazo_c_par_AR, sazo_c_par_I, sazo_c_par_MA, sazo_c_par_n = sazo_c_par_AR_.empty(), sazo_c_par_I_.empty(), sazo_c_par_MA_.empty(), sazo_c_par_n_.empty()
        par_ARIMA["P"] = sazo_c_par_AR.number_input('P (Sazonal):', min_value=0, max_value=1000, value=0, step=1, format = "%d", key = "P")
        par_ARIMA["D"] = sazo_c_par_I.number_input('D (Sazonal):', min_value=0, max_value=1000, value=0, step=1, format = "%d", key = "D")
        par_ARIMA["Q"] = sazo_c_par_MA.number_input('Q (Sazonal):', min_value=0, max_value=1000, value=0, step=1, format = "%d", key = "Q")
        par_ARIMA["n"] = sazo_c_par_n.number_input('Período da sazonalidade (n):', min_value=0, max_value=1000, value=0, step=1, format = "%d", key = "n")

        if par_ARIMA["n"] == 1:
            st.markdown("A sazonalidade precisa ser maior do que 1 ou nula")
        elif par_ARIMA["n"] == 0 and (par_ARIMA["P"] > 0 or par_ARIMA["D"] > 0 or par_ARIMA["Q"] > 0):
            st.markdown("Para utilização dos argumentos sazonais (P, D, Q), a sazonalidade deve ser maior do que 1.")
        elif par_ARIMA["trend"] == 'c' and (par_ARIMA["d"] > 0 or par_ARIMA["D"] > 0):
            st.markdown("O modelo com constante não pode ser diferenciado.")
        else:
            param_models = st.empty()
            param_models_text(param_models, par_ARIMA)


            ##################################################
            # auto.arima
            ##################################################
            st.markdown('''Você pode utilizar um método automático para escolher os parâmetros do modelo. 
            Utilizamos a biblioteca [pmdarima](https://alkaline-ml.com/pmdarima/), que é baseada na função auto.arima 
            do pacote forecast da linguagem R.  
            Se a série for sazonal você deve configurar acima a sazonalidade e depois executar o método de seleção.''')

            if st.button('Estimar com auto.arima'):
                if par_ARIMA["n"] > 0:
                    model_auto_arima = pm.auto_arima(y, seasonal=True, m=par_ARIMA["n"])
                else:
                    model_auto_arima = pm.auto_arima(y, seasonal=False)
                par_ARIMA["p"] = model_auto_arima.get_params()["order"][0]
                par_ARIMA["d"] = model_auto_arima.get_params()["order"][1]
                par_ARIMA["q"] = model_auto_arima.get_params()["order"][2]

                par_ARIMA["P"] = model_auto_arima.get_params()["seasonal_order"][0]
                par_ARIMA["D"] = model_auto_arima.get_params()["seasonal_order"][1]
                par_ARIMA["Q"] = model_auto_arima.get_params()["seasonal_order"][2]
                par_ARIMA["n"] = model_auto_arima.get_params()["seasonal_order"][3]

                c_par_AR.number_input('p:', min_value=0, max_value=1000, value=par_ARIMA["p"], step=1, format = "%d", key = "pn")
                c_par_I.number_input('d:', min_value=0, max_value=1000, value=par_ARIMA["d"], step=1, format = "%d", key = "dn")
                c_par_MA.number_input('q:', min_value=0, max_value=1000, value=par_ARIMA["q"], step=1, format = "%d", key = "qn")

                sazo_c_par_AR.number_input('P (Sazonal):', min_value=0, max_value=1000, value=par_ARIMA["P"], step=1, format = "%d", key = "Pn")
                sazo_c_par_I.number_input('D (Sazonal):', min_value=0, max_value=1000, value=par_ARIMA["D"], step=1, format = "%d", key = "Dn")
                sazo_c_par_MA.number_input('Q (Sazonal):', min_value=0, max_value=1000, value=par_ARIMA["Q"], step=1, format = "%d", key = "Qn")
                sazo_c_par_n.number_input('Período da sazonalidade (n):', min_value=0, max_value=1000, value=par_ARIMA["n"], step=1, format = "%d", key = "nn")

                param_models_text(param_models, par_ARIMA)

                st.text("Parâmetros estimados com auto.arima. Foi considerado uma sazonalidade {}".format(par_ARIMA["n"]))

            ##################################################
            # Refaz os testes ADF e ACF/PACF caso exista diferenciação
            ##################################################
            if par_ARIMA["d"] > 0:
                tests_adf_autocorr(y.diff(periods=par_ARIMA["d"])[par_ARIMA["d"]:], 'Testes estatísticos para a série com d = {}'.format(par_ARIMA["d"]))


            ##################################################
            # Treina modelo
            ##################################################
            model = ARIMA(df.set_index("DATA"), order=(par_ARIMA["p"], par_ARIMA["d"], par_ARIMA["q"]), 
                        seasonal_order=(par_ARIMA["P"], par_ARIMA["D"], par_ARIMA["Q"], par_ARIMA["n"]), trend = par_ARIMA["trend"])
            model_fit = model.fit()
            y_pred = model_fit.forecast(steps=60)
            
            model_summary = st.beta_expander('Veja os parâmetros do modelo:')
            with model_summary:
                st.write(model_fit.summary())

            ##################################################
            # Plota Forecast
            ##################################################
            fig_forecast = plot_grafico_projecao(df, y_pred.reset_index(), nome_data, nome_qty)
            st.plotly_chart(fig_forecast)

            ##################################################
            # Análise dos resíduos
            ##################################################
            resid_analise = st.beta_expander('Análise dos resíduos:')
            with resid_analise:
                
                residuos = model_fit.resid[1:]
                st.markdown('**Funções de autocorrelação dos resíduos:**')
                c_acf_resi, c_pacf_resi = st.beta_columns(2)
                acf_resi = ts.acf(residuos, nlags = 25, alpha = .05)
                c_acf_resi.plotly_chart(plot_auto_correlation(acf_resi, 25, "ACF dos resíduos"))
                pacf_resi = ts.pacf(residuos, nlags = 25, alpha = .05)
                c_pacf_resi.plotly_chart(plot_auto_correlation(pacf_resi, 25, "PACF dos resíduos"))
                
                lb_results = sm.stats.diagnostic.acorr_ljungbox(model_fit.resid.values, lags=1, return_df = True)["lb_pvalue"].values[0]
                if lb_results > 0.05:
                    text_result_lb = (f'- **Não existe correlação serial do lag 1** 👍. P-Valor: {lb_results:.3f}')
                else:
                    text_result_lb = (f'- **Existe correlação serial do lag 1** 👎. P-Valor: {lb_results:.3f}')

                st.markdown('''
                **Teste de Ljung–Box para autocorrelação.** 
                - **Hipóteses:**
                    - Hipótese nula ($H_0$): Não existe correlação serial de ordem até "p" (que é escolhido por quem está utilizando o teste);  
                    - Hipótese alternativa ($H_1$): Existe correlação serial de ao menos uma ordem até "p".  
                ''' + text_result_lb)




def plot_grafico_projecao(X_train, X_pred, nome_data, nome_qty):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(X_train[nome_data]), y=list(X_train[nome_qty]), name = "Observações"))
    fig.add_trace(go.Scatter(x=X_pred["index"], y=X_pred["predicted_mean"], name = "Projeção"))
    fig.update_layout()

    fig.update_layout(
        title_text="Gráfico Observações e Projeção",
        width=1200, height=400,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m",  step="month", stepmode="backward"),
                    dict(count=6, label="6m",  step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year",  stepmode="todate"),
                    dict(count=1, label="1y",  step="year",  stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig

def tests_adf_autocorr(y, texto):
    statistical_tests = st.beta_expander(texto)
    with statistical_tests:
        # Teste de estacionariedade
        st.markdown('***')
        st.markdown('**Teste de estacionariedade de Dickey-Fuller Aumentado (ADF):**')
        check_adfuller(y)
        
        # Plota ACF e PACF
        st.markdown('***')
        c_acf, c_pacf = st.beta_columns(2)
        acf = ts.acf(y, nlags = 25, alpha = .05)
        c_acf.plotly_chart(plot_auto_correlation(acf, 25, "ACF"))
        pacf = ts.pacf(y, nlags = 25, alpha = .05)
        c_pacf.plotly_chart(plot_auto_correlation(pacf, 25, "PACF"))

def param_models_text(param_models, par_ARIMA):
    param_models.markdown(f'''
    Modelo sendo utilizado (notação $(p, d, q)x(P, D, Q, s)$):
    $({par_ARIMA["p"]}, {par_ARIMA["d"]}, {par_ARIMA["q"]})x({par_ARIMA["P"]}, {par_ARIMA["D"]}, {par_ARIMA["Q"]}, {par_ARIMA["n"]})$
    ''')


def check_adfuller(y):
    est = ts.adfuller(y)[1]
    if est <= 0.05:
        st.markdown("👍 A série **é estacionária** com p-valor de : {:0.4f}".format(est))
    else:
        st.markdown("👎 A série **não é estacionária** com p-valor de : {:0.4f}".format(est))

@st.cache
def plot_auto_correlation(serie, n, versao):
    x=list(range(0, n+1))
    y=serie[0]
    y_upper = serie[1][:,1] - y
    y_lower = serie[1][:,0] - y

    fig = go.Figure([
        go.Scatter(x=x, y=y, line_color='red', mode='markers'),
        go.Bar(x=x, y=y, width  = 0.1),
        go.Scatter(x=x, y=y_lower, fill='tonexty', mode='none', line_color='indigo'),
        go.Scatter(x=x, y=y_upper, fill='tonexty', mode='none', line_color='indigo')
    ])
    fig.update_layout(showlegend=False, title=versao, xaxis_title="Lags", yaxis_title="Corr",  width=500, height=400)
    return fig

@st.cache
def plot_grafico_1(df, nome_data, nome_qty):
    # Based in: https://plotly.com/python/range-slider/

    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(df[nome_data]), y=list(df[nome_qty])))

    # Set title
    fig.update_layout(title_text="Gráfico da série importada")

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m",  step="month", stepmode="backward"),
                    dict(count=6, label="6m",  step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year",  stepmode="todate"),
                    dict(count=1, label="1y",  step="year",  stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig


if __name__ == "__main__":
    main()
    