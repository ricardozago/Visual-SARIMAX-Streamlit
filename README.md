Ferramenta com interface gráfica para auxiliar o desenvolvimento de modelos SARIMAX, seguindo a metodologia de Box–Jenkins, permitindo a identificação do modelo, com funções de autocorrelação e autocorrelação parcial, estimação dos parâmetros e checagem das premissas estatísticas (teste de autocorrelação dos resíduos). Além do uso de variáveis exógenas ao modelo.

Estamos muito acostumados a utilizar Jupyter notebooks para análise exploratória e desenvolvimento de modelos, mas as vezes queremos alguma solução mais simples e rápida.

Também existe o caso de pessoas que estão acostumadas a utilizar o Excel e ferramentas como o seu toolbox de análise de dados, mas não sabem programar. O Visual SARIMAX visa atender essas pessoas.

Esta ferramenta utiliza as bibliotecas:
- Streamlit;
- Pmdarima;
- Numpy;
- Pandas;
- Statsmodels;
- Sklearn;
- Plotly.

Inicialmente deve-se importar um dataset, utilizamos como exemplos os datasets:
- [AirPassengers](https://www.kaggle.com/chirag19/air-passengers) para modelos SARIMA;
- [Icecream](https://www3.nd.edu/~busiforc/handouts/Data%20and%20Stories/regression/ice%20cream%20consumption/icecream.html) para modelos com exógenas.

**Importando um dataset, o nome dos campos e data e do valor da série temporal é fornecido ao programa, se existir outras colunas elas serão consideradas variáveis exógenas.**

![01 - Importação.png](https://github.com/ricardozago/Streamlit_SARIMAX/blob/main/Imagens/01%20-%20Importa%C3%A7%C3%A3o.png)

**Escolhendo o modelo SARIMA (é possível utilizar o auto.arima do pacote pmdarima!):**

![02 - Parâmetros.png](https://github.com/ricardozago/Streamlit_SARIMAX/blob/main/Imagens/02%20-%20Par%C3%A2metros.png)

**Verificando a projeção e se os resíduos estão coerentes (conforme metodologia Box–Jenkins):**

![03 - Projeção e testes.png](https://github.com/ricardozago/Streamlit_SARIMAX/blob/main/Imagens/03%20-%20Proje%C3%A7%C3%A3o%20e%20testes.png)

Estamos abertos a contribuições!
