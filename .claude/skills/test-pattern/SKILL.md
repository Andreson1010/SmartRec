# Skill: test-pattern

Padrão para escrever testes no SmartRec.

## Convenções gerais

- Arquivos em `tests/` espelhando a estrutura do projeto
  (`tests/ml/collaborative/`, `tests/api/`, etc.)
- Dados sintéticos gerados nos fixtures — nunca depender de arquivos reais
- Cobrir: happy path, edge cases e erros esperados
- Mockar toda dependência externa (MLflow, disco, modelos treinados)

## Template

```python
"""
tests/<modulo>/test_<componente>.py
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures — dados sintéticos reutilizáveis
# ---------------------------------------------------------------------------

@pytest.fixture()
def interactions_df() -> pd.DataFrame:
    """DataFrame mínimo de interações para treino/teste."""
    rng = np.random.default_rng(42)

    n_users    = 20
    n_products = 30
    n_rows     = 200

    return pd.DataFrame({
        "user_id":    [f"u{i}" for i in rng.integers(0, n_users,    n_rows)],
        "product_id": [f"p{i}" for i in rng.integers(0, n_products, n_rows)],
        "rating":     rng.integers(1, 6, n_rows).astype("float32"),
        "timestamp":  rng.integers(1_600_000_000, 1_700_000_000, n_rows),
    })


@pytest.fixture()
def trained_model(interactions_df: pd.DataFrame):
    """Instância do modelo já treinada com dados sintéticos."""
    from ml.<submodule>.<model> import MyModel

    with patch("mlflow.start_run"), patch("mlflow.log_param"), patch("mlflow.log_metrics"):
        model = MyModel(param_a=5).fit(interactions_df)

    return model


@pytest.fixture()
def tmp_model_dir(tmp_path: Path) -> Path:
    """Diretório temporário para serialização do modelo."""
    return tmp_path / "model"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestMyModelFit:
    def test_fit_returns_self(self, interactions_df: pd.DataFrame) -> None:
        from ml.<submodule>.<model> import MyModel

        with patch("mlflow.start_run"), patch("mlflow.log_param"), patch("mlflow.log_metrics"):
            model = MyModel()
            result = model.fit(interactions_df)

        assert result is model

    def test_fit_marks_as_fitted(self, interactions_df: pd.DataFrame) -> None:
        from ml.<submodule>.<model> import MyModel

        with patch("mlflow.start_run"), patch("mlflow.log_param"), patch("mlflow.log_metrics"):
            model = MyModel().fit(interactions_df)

        assert model._is_fitted is True


class TestMyModelPredict:
    def test_predict_returns_list(self, trained_model, interactions_df: pd.DataFrame) -> None:
        user_id = interactions_df["user_id"].iloc[0]
        result = trained_model.predict(user_id, top_k=5)

        assert isinstance(result, list)
        assert len(result) <= 5

    def test_predict_item_schema(self, trained_model, interactions_df: pd.DataFrame) -> None:
        user_id = interactions_df["user_id"].iloc[0]
        items = trained_model.predict(user_id, top_k=3)

        for item in items:
            assert "product_id" in item
            assert "score" in item
            assert 0.0 <= item["score"] <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_unknown_user_returns_fallback(self, trained_model) -> None:
        """Usuário nunca visto deve receber recomendações populares."""
        result = trained_model.predict("user_never_seen", top_k=10)

        assert isinstance(result, list)    # não lança exceção
        assert len(result) > 0             # retorna algo

    def test_top_k_zero_raises(self, trained_model, interactions_df: pd.DataFrame) -> None:
        user_id = interactions_df["user_id"].iloc[0]

        with pytest.raises((ValueError, Exception)):
            trained_model.predict(user_id, top_k=0)

    def test_empty_dataframe_raises_on_fit(self) -> None:
        from ml.<submodule>.<model> import MyModel

        empty = pd.DataFrame(columns=["user_id", "product_id", "rating"])

        with pytest.raises(Exception):
            with patch("mlflow.start_run"), patch("mlflow.log_param"), patch("mlflow.log_metrics"):
                MyModel().fit(empty)


# ---------------------------------------------------------------------------
# Serialização
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, trained_model, tmp_model_dir: Path) -> None:
        from ml.<submodule>.<model> import MyModel

        trained_model.save(tmp_model_dir)
        loaded = MyModel.load(tmp_model_dir)

        assert loaded._is_fitted is True

    def test_save_creates_directory(self, trained_model, tmp_model_dir: Path) -> None:
        trained_model.save(tmp_model_dir)

        assert tmp_model_dir.exists()


# ---------------------------------------------------------------------------
# Mock de dependências externas
# ---------------------------------------------------------------------------

class TestMLflowLogging:
    def test_fit_logs_params_and_metrics(self, interactions_df: pd.DataFrame) -> None:
        from ml.<submodule>.<model> import MyModel

        mock_run = MagicMock()

        with (
            patch("mlflow.start_run", return_value=mock_run),
            patch("mlflow.log_param")  as mock_log_param,
            patch("mlflow.log_metrics") as mock_log_metrics,
        ):
            MyModel(param_a=7).fit(interactions_df)

        mock_log_param.assert_called_with("param_a", 7)
        mock_log_metrics.assert_called_once()
```

## Checklist

- [ ] Fixtures geram dados sintéticos com `numpy.random.default_rng(seed)` — resultados determinísticos
- [ ] Fixture `trained_model` mocka MLflow para não exigir servidor rodando
- [ ] Happy path: resultado correto com entrada válida
- [ ] Edge cases: cold start, top_k inválido, DataFrame vazio
- [ ] Serialização: `save` + `load` preserva estado (`_is_fitted`)
- [ ] Dependências externas (MLflow, disco, HTTP) sempre mockadas com `unittest.mock.patch`
- [ ] Nomes de teste descrevem o comportamento esperado, não a implementação
- [ ] Sem `time.sleep`, sem I/O real em disco além de `tmp_path`
