import importlib
import os
import sys
import types
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture()
def utils_setup(monkeypatch):
    logs = []

    class DummyExpander:
        def markdown(self, *args, **kwargs):
            logs.append(args[0])
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass

    class DummyColumn:
        def plotly_chart(self, *a, **k):
            pass

    st = types.SimpleNamespace()
    class DummySessionState(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    st.session_state = DummySessionState()
    st.expander = lambda *a, **k: DummyExpander()
    st.markdown = lambda *a, **k: logs.append(a[0])
    st.download_button = lambda *a, **k: None
    st.columns = lambda n: [DummyColumn() for _ in range(n)]
    st.cache_data = lambda func=None, **kw: (lambda *a, **k: func(*a, **k)) if func else (lambda f: f)
    st.write = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'streamlit', st)

    class DummyTrace:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
    class DummyFigure:
        def __init__(self, data=None):
            self.data = list(data or [])
            self.layout = {}
        def add_trace(self, trace):
            self.data.append(trace)
        def update_layout(self, **kw):
            self.layout.update(kw)
    go = types.SimpleNamespace(Figure=DummyFigure, Scatter=DummyTrace, Bar=DummyTrace)
    monkeypatch.setitem(sys.modules, 'plotly', types.SimpleNamespace(graph_objects=go))
    monkeypatch.setitem(sys.modules, 'plotly.graph_objects', go)

    ts = types.SimpleNamespace(
        acf=lambda y, nlags, alpha: ([0]*(nlags+1), [[0, 0]]*(nlags+1)),
        pacf=lambda y, nlags, alpha: ([0]*(nlags+1), [[0, 0]]*(nlags+1)),
        adfuller=lambda y: [None, 0.5],
    )
    monkeypatch.setitem(sys.modules, 'statsmodels', types.SimpleNamespace(tsa=types.SimpleNamespace(stattools=ts)))
    monkeypatch.setitem(sys.modules, 'statsmodels.tsa.stattools', ts)

    class DummyDataFrame(dict):
        def __init__(self, data=None, **kwargs):
            super().__init__(data or {})

        def to_csv(self, index=False):
            import io, csv
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(self.keys())
            for row in zip(*self.values()):
                writer.writerow(row)
            return output.getvalue()

        def reset_index(self):
            self['index'] = list(range(len(next(iter(self.values()), []))))
            return self

        @property
        def empty(self):
            return not any(self.values())

    def read_csv(path):
        import csv
        data = {}
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k, v in row.items():
                    data.setdefault(k, []).append(v)
        return DummyDataFrame(data)

    pd = types.SimpleNamespace(DataFrame=DummyDataFrame, read_csv=read_csv)
    monkeypatch.setitem(sys.modules, 'pandas', pd)

    utils = importlib.import_module('utils')
    plots = importlib.import_module('plots')
    importlib.reload(utils)
    importlib.reload(plots)
    return utils, plots, logs, ts, st


def test_param_models_text(utils_setup):
    utils, _, logs, _, _ = utils_setup
    logs.clear()
    dummy = types.SimpleNamespace(markdown=lambda text: logs.append(text))
    params = {'p':1,'d':0,'q':1,'P':0,'D':1,'Q':0,'n':12}
    utils.param_models_text(dummy, params)
    assert any("(1, 0, 1)" in m for m in logs)


def test_download_dataframe(utils_setup):
    utils, _, _, _, st = utils_setup
    calls = []
    st.download_button = lambda *a, **k: calls.append((a, k))
    pd = sys.modules['pandas']
    df = pd.DataFrame({'A':[1,2]})
    utils.download_dataframe(df, file_name="out.csv")
    assert calls and calls[0][1]['file_name'] == "out.csv"


def test_plot_grafico_1(utils_setup):
    _, plots, _, _, _ = utils_setup
    pd = sys.modules['pandas']
    df = pd.DataFrame({'DATA':[1,2], 'QTY':[3,4]})
    fig = plots.plot_grafico_1(df, 'DATA', 'QTY', is_data=False)
    assert fig.layout.get('title_text') == 'Gráfico da série importada'
    assert len(fig.data) == 1


def test_plot_grafico_projecao(utils_setup):
    _, plots, _, _, _ = utils_setup
    pd = sys.modules['pandas']
    X_train = pd.DataFrame({'DATA':[1,2], 'QTY':[1,2]})
    X_test = pd.DataFrame({'DATA':[3], 'QTY':[3]})
    X_pred = pd.DataFrame({'predicted_mean':[4,5]}, index=[0,1])
    fig = plots.plot_grafico_projecao(X_train, X_test, X_pred, 'DATA', 'QTY', is_data=False)
    assert len(fig.data) == 3
    fig2 = plots.plot_grafico_projecao(X_train, None, X_pred, 'DATA', 'QTY', is_data=False)
    assert len(fig2.data) == 2
