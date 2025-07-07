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

    # Mock streamlit
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

    # Mock plotly
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

    # Mock statsmodels.tsa.stattools
    ts = types.SimpleNamespace(
        acf=lambda y, nlags, alpha: ([0]*(nlags+1), [[0, 0]]*(nlags+1)),
        pacf=lambda y, nlags, alpha: ([0]*(nlags+1), [[0, 0]]*(nlags+1)),
        adfuller=lambda y: [None, 0.5],
    )
    monkeypatch.setitem(sys.modules, 'statsmodels', types.SimpleNamespace(tsa=types.SimpleNamespace(stattools=ts)))
    monkeypatch.setitem(sys.modules, 'statsmodels.tsa.stattools', ts)

    # Minimal pandas replacement
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

    utils = importlib.import_module('visual_sarimax.utils')
    plots = importlib.import_module('visual_sarimax.plots')
    importlib.reload(utils)
    importlib.reload(plots)
    return utils, plots, logs, ts, st


def test_set_session_state(utils_setup):
    utils, _, _, _, st = utils_setup
    st.session_state.clear()
    utils.set_session_state()
    assert st.session_state['df'] is None
    assert st.session_state['nome_data'] == 'DATA'
    assert st.session_state['nome_qty'] == 'QTY'
    assert st.session_state['is_data'] is True


def test_plot_auto_correlation(utils_setup):
    _, plots, _, _, _ = utils_setup
    class NumArray(list):
        def __sub__(self, other):
            return NumArray([a - b for a, b in zip(self, other)])

    class Matrix(list):
        def __getitem__(self, idx):
            if isinstance(idx, tuple):
                rows, col = idx
                if isinstance(rows, slice):
                    return NumArray([row[col] for row in super().__getitem__(rows)])
                return super().__getitem__(rows)[col]
            return super().__getitem__(idx)

    serie = ([0, 0, 0], Matrix([[0, 0], [0, 0], [0, 0]]))
    fig = plots.plot_auto_correlation(serie, 2, 'ACF')
    assert len(fig.data) == 4
    assert fig.layout['title'] == 'ACF'


def test_check_adfuller_stationary(utils_setup):
    utils, _, logs, ts, _ = utils_setup
    logs.clear()
    ts.adfuller = lambda y: [None, 0.01]
    utils.check_adfuller([1, 2, 3])
    assert any('é estacionária' in m for m in logs)


def test_check_adfuller_non_stationary(utils_setup):
    utils, _, logs, ts, _ = utils_setup
    logs.clear()
    ts.adfuller = lambda y: [None, 0.1]
    utils.check_adfuller([1, 2, 3])
    assert any('não é estacionária' in m for m in logs)


def test_load_dataset(utils_setup):
    utils, _, _, _, _ = utils_setup
    df = utils.load_dataset("AirPassengers")
    assert not df.empty
