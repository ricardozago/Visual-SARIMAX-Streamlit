# Visual SARIMAX

Ferramenta com interface gráfica para auxiliar o desenvolvimento de modelos SARIMAX seguindo a metodologia de Box--Jenkins. O aplicativo permite:

- Visualização das funções de autocorrelação e autocorrelação parcial;
- Estimação dos parâmetros do modelo e validação de hipóteses estatísticas (ADF e Ljung--Box);
- Utilização opcional de variáveis exógenas;
- Download das projeções geradas.

A interface foi pensada para usuários que nem sempre estão familiarizados com código Python, sendo uma alternativa aos tradicionais notebooks.

## Exemplos de uso

Alguns datasets de demonstração estão disponíveis no repositório:

- [AirPassengers](https://www.kaggle.com/chirag19/air-passengers)
- [Icecream](https://www3.nd.edu/~busiforc/handouts/Data%20and%20Stories/regression/ice%20cream%20consumption/icecream.html)

Para utilizá‑los selecione a opção desejada na própria aplicação ou importe um arquivo CSV/XLS(X) com as colunas de data e valor da série temporal.

Os arquivos de demonstração também podem ser carregados via código:

```python
from visual_sarimax import load_dataset
df = load_dataset("AirPassengers")
```

![Importação](https://github.com/ricardozago/Streamlit_SARIMAX/blob/main/Imagens/01%20-%20Importa%C3%A7%C3%A3o.png)

Após escolher os parâmetros SARIMA (ou utilizar o `auto.arima`) visualize a projeção e os testes de diagnóstico:

![Projeção e testes](https://github.com/ricardozago/Streamlit_SARIMAX/blob/main/Imagens/03%20-%20Proje%C3%A7%C3%A3o%20e%20testes.png)

## Instalação

Instale a aplicação em modo desenvolvimento com:

```bash
pip install -e .
```

Em seguida execute o aplicativo com:

```bash
streamlit run -m visual_sarimax.app
```

## Desenvolvimento

Clone o projeto e instale as dependências listadas em `requirements.txt`. Para rodar os testes automatizados utilize:

```bash
pytest -q
```

Contribuições são bem-vindas!

---

## English summary

**Visual SARIMAX** is a Streamlit application that guides you through the Box--Jenkins methodology for SARIMAX models. Import a dataset, choose the model parameters (or rely on `auto.arima`) and inspect the forecast along with diagnostic tests. Run it locally with `streamlit run -m visual_sarimax.app` after installing the package in editable mode.
