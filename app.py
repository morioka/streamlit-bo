from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from io import BytesIO

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "work", ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class ModelBundle:
    model: Pipeline
    feature_columns: list[str]
    target_column: str
    cv_prediction: np.ndarray
    train_prediction: np.ndarray
    selected_kernel: str
    cv_metrics: dict[str, float]
    train_metrics: dict[str, float]


def main() -> None:
    st.set_page_config(
        page_title="実験点提案@ベイズ最適化",
        layout="wide",
        page_icon="random",
    )
    st.title("実験点提案@ベイズ最適化")

    input_format = st.radio("入力形式", ["Excel", "CSV", "TSV"], horizontal=True)
    data_file = st.file_uploader(
        "実験データを読み込んで下さい。",
        type=file_types_for_format(input_format),
        key="train_data",
    )

    limit_file = None
    if input_format != "Excel":
        limit_file = st.file_uploader(
            "探索範囲を読み込んで下さい。",
            type=file_types_for_format(input_format),
            key="limit_data",
        )

    if data_file is None or (input_format != "Excel" and limit_file is None):
        st.info("Excelなら `sample_data.xlsx`、CSV/TSVなら実験データと探索範囲をそれぞれアップロードしてください。")
        return

    try:
        data, limit_data = load_training_inputs(data_file, limit_file, input_format)
    except Exception as exc:
        st.error(f"データの読み込みに失敗しました: {exc}")
        return

    if data.empty or limit_data.empty:
        st.error("実験データと探索範囲の両方にデータを入れてください。")
        return

    tabs_set(data, limit_data)


def file_types_for_format(input_format: str) -> list[str]:
    return {"Excel": ["xlsx"], "CSV": ["csv"], "TSV": ["tsv", "txt"]}[input_format]


def load_training_inputs(data_file, limit_file, input_format: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if input_format == "Excel":
        return (
            pd.read_excel(data_file, sheet_name=0),
            pd.read_excel(data_file, sheet_name=1, index_col=0),
        )
    return read_table(data_file, input_format), read_table(limit_file, input_format, index_col=0)


def read_table(uploaded_file, input_format: str, index_col: int | None = None) -> pd.DataFrame:
    separator = "," if input_format == "CSV" else "\t"
    return pd.read_csv(uploaded_file, sep=separator, index_col=index_col)


def tabs_set(data: pd.DataFrame, limit_data: pd.DataFrame) -> None:
    tabs = st.tabs(["データの確認", "実験点の提案", "解析", "予測"])
    ss = st.session_state

    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    default_target_index = max(len(numeric_columns) - 1, 0)

    with tabs[0]:
        st.subheader("実験データ")
        st.dataframe(data, use_container_width=True)
        st.subheader("探索範囲")
        st.dataframe(limit_data, use_container_width=True)
        scatter_controls(data)

    with tabs[1]:
        if not numeric_columns:
            st.warning("数値列が必要です。")
        else:
            target_column = st.selectbox(
                "目的変数",
                numeric_columns,
                index=default_target_index,
                key="target_column",
            )
            feature_columns = st.multiselect(
                "説明変数",
                [col for col in numeric_columns if col != target_column],
                default=[col for col in numeric_columns if col != target_column],
                key="feature_columns",
            )
            objective = st.radio("目的", ["最大化", "最小化"], horizontal=True)
            acquisition = st.selectbox("獲得関数", ["EI", "PI", "MI"], index=0)
            n_suggestions = st.number_input("提案する実験点数", 1, 50, 10, 1)
            max_candidates = st.number_input("候補点の上限", 500, 100_000, 20_000, 500)

            cache_key = optimization_cache_key(
                data,
                limit_data,
                target_column,
                feature_columns,
                objective,
                acquisition,
                n_suggestions,
                max_candidates,
            )

            if st.checkbox("実験点の提案"):
                if ss.get("optimization_cache_key") != cache_key:
                    with st.spinner("ガウス過程回帰モデルを構築し、次の実験点を探索しています..."):
                        try:
                            bundle, next_samples, candidates = run_optimization(
                                data=data,
                                limit_data=limit_data,
                                target_column=target_column,
                                feature_columns=feature_columns,
                                objective=objective,
                                acquisition=acquisition,
                                n_suggestions=int(n_suggestions),
                                max_candidates=int(max_candidates),
                            )
                        except Exception as exc:
                            st.error(f"実験点の提案に失敗しました: {exc}")
                            return
                    ss["optimization_cache_key"] = cache_key
                    ss["model_bundle"] = bundle
                    ss["next_samples"] = next_samples
                    ss["candidates"] = candidates

                bundle = ss["model_bundle"]
                next_samples = ss["next_samples"]
                display_model_results(data, bundle)
                st.subheader("提案された実験点")
                st.dataframe(next_samples, use_container_width=True)
                download_dataframe(next_samples, "提案結果を保存", "suggested_experiments")

    with tabs[2]:
        if "model_bundle" not in ss:
            st.info("先に「実験点の提案」タブでモデルを構築してください。")
        elif st.checkbox("SHAP解析", help="モデルを構築してから実行できます"):
            with st.spinner("SHAP値を計算しています..."):
                plot_path = shap_explain(ss["model_bundle"], data)
            if plot_path is None:
                st.warning("SHAPが利用できません。`pip install shap` 後に再実行してください。")
            else:
                st.image(plot_path)

    with tabs[3]:
        if "model_bundle" not in ss:
            st.info("先に「実験点の提案」タブでモデルを構築してください。")
        else:
            prediction_tab(ss["model_bundle"])


def scatter_controls(data: pd.DataFrame) -> None:
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_columns) < 2:
        return

    st.subheader("散布図")
    cols = st.columns(4)
    x_col = cols[0].selectbox("X軸", numeric_columns, index=0)
    y_col = cols[1].selectbox("Y軸", numeric_columns, index=min(1, len(numeric_columns) - 1))
    z_col = cols[2].selectbox("Z軸", ["なし"] + numeric_columns)
    color_col = cols[3].selectbox("色", ["なし"] + numeric_columns)

    if z_col == "なし":
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=None if color_col == "なし" else color_col,
            hover_data=data.columns,
        )
    else:
        fig = px.scatter_3d(
            data,
            x=x_col,
            y=y_col,
            z=z_col,
            color=None if color_col == "なし" else color_col,
            hover_data=data.columns,
        )
    st.plotly_chart(fig, use_container_width=True)


def run_optimization(
    data: pd.DataFrame,
    limit_data: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    objective: str,
    acquisition: str,
    n_suggestions: int,
    max_candidates: int,
) -> tuple[ModelBundle, pd.DataFrame, pd.DataFrame]:
    if not feature_columns:
        raise ValueError("説明変数を1つ以上選択してください。")

    clean_data = data[feature_columns + [target_column]].copy()
    clean_data = clean_data.dropna(subset=[target_column])
    if clean_data.empty:
        raise ValueError("目的変数に有効な値がありません。")

    x = clean_data[feature_columns]
    y = clean_data[target_column].to_numpy(dtype=float)
    direction = 1 if objective == "最大化" else -1

    base_bundle = build_best_model(x, y, feature_columns, target_column)
    candidates = sample_generation(limit_data, x, feature_columns, max_candidates)
    next_samples = suggest_samples(
        x=x,
        y=y,
        template_model=base_bundle.model,
        candidates=candidates,
        feature_columns=feature_columns,
        target_column=target_column,
        direction=direction,
        acquisition=acquisition,
        n_suggestions=n_suggestions,
    )
    return base_bundle, next_samples, candidates


def build_best_model(
    x: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list[str],
    target_column: str,
) -> ModelBundle:
    kernels = kernel_candidates(x.shape[1])
    cv = cv_strategy(len(x))
    best_score = -np.inf
    best_model: Pipeline | None = None
    best_cv_pred: np.ndarray | None = None
    best_kernel_name = ""

    for kernel in kernels:
        model = make_model(kernel)
        try:
            if cv is None:
                cv_pred = np.repeat(np.mean(y), len(y))
                score = -mean_squared_error(y, cv_pred)
            else:
                cv_pred = cross_val_predict(model, x, y, cv=cv, n_jobs=None)
                score = r2_score(y, cv_pred) if len(np.unique(y)) > 1 else -mean_squared_error(y, cv_pred)
            if score > best_score:
                best_score = score
                best_model = clone(model).fit(x, y)
                best_cv_pred = cv_pred
                best_kernel_name = str(kernel)
        except Exception:
            continue

    if best_model is None or best_cv_pred is None:
        fallback = make_model(ConstantKernel(1.0) * RBF(length_scale=np.ones(x.shape[1])) + WhiteKernel())
        best_model = fallback.fit(x, y)
        best_cv_pred = np.repeat(np.mean(y), len(y))
        best_kernel_name = "fallback RBF"

    train_prediction = best_model.predict(x)
    return ModelBundle(
        model=best_model,
        feature_columns=feature_columns,
        target_column=target_column,
        cv_prediction=best_cv_pred,
        train_prediction=train_prediction,
        selected_kernel=best_kernel_name,
        cv_metrics=metrics(y, best_cv_pred),
        train_metrics=metrics(y, train_prediction),
    )


def make_model(kernel) -> Pipeline:
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("gpr", gpr),
        ]
    )


def kernel_candidates(n_features: int) -> list:
    length_scale = np.ones(n_features)
    base = ConstantKernel(1.0, (1e-3, 1e3))
    noise = WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e1))
    return [
        base * RBF(length_scale=length_scale) + noise,
        base * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) + noise,
        base * Matern(length_scale=length_scale, nu=0.5) + noise,
        base * Matern(length_scale=length_scale, nu=1.5) + noise,
        base * Matern(length_scale=length_scale, nu=2.5) + noise,
        base * RationalQuadratic(alpha=0.1) + noise,
        base * RationalQuadratic(alpha=1.0) + noise,
        base * RationalQuadratic(alpha=10.0) + noise,
        base * DotProduct() + noise,
        base * (DotProduct() + RBF(length_scale=length_scale)) + noise,
        base * ExpSineSquared(length_scale=1.0, periodicity=1.0) + noise,
    ]


def cv_strategy(n_samples: int):
    if n_samples < 3:
        return None
    if n_samples <= 10:
        return LeaveOneOut()
    return KFold(n_splits=min(5, n_samples), shuffle=True, random_state=RANDOM_STATE)


def sample_generation(
    limit_data: pd.DataFrame,
    observed_x: pd.DataFrame,
    feature_columns: list[str],
    max_candidates: int,
) -> pd.DataFrame:
    specs = normalize_limit_data(limit_data)
    grids: list[np.ndarray] = []
    rng = np.random.default_rng(RANDOM_STATE)

    total_grid_size = 1
    for col in feature_columns:
        values = candidate_values_for_column(col, specs, observed_x[col])
        grids.append(values)
        total_grid_size *= len(values)

    if total_grid_size <= max_candidates:
        mesh = np.meshgrid(*grids, indexing="ij")
        arr = np.column_stack([m.ravel() for m in mesh])
    else:
        arr = np.column_stack([rng.choice(values, size=max_candidates, replace=True) for values in grids])
        arr = np.unique(arr, axis=0)

    candidates = pd.DataFrame(arr, columns=feature_columns)
    candidates = candidates.drop_duplicates(ignore_index=True)
    return candidates


def normalize_limit_data(limit_data: pd.DataFrame) -> pd.DataFrame:
    normalized = limit_data.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    normalized.index = normalized.index.map(str)
    return normalized


def candidate_values_for_column(col: str, specs: pd.DataFrame, observed: pd.Series) -> np.ndarray:
    observed_numeric = pd.to_numeric(observed, errors="coerce").dropna()
    if col in specs.index:
        row = specs.loc[col]
        min_value = read_number(row, ["min", "lower", "下限"], observed_numeric.min())
        max_value = read_number(row, ["max", "upper", "上限"], observed_numeric.max())
        step = read_number(row, ["step", "刻み"], np.nan)
        points = int(read_number(row, ["points", "num", "候補数"], 25))
    else:
        min_value = float(observed_numeric.min())
        max_value = float(observed_numeric.max())
        step = np.nan
        points = 25

    if not np.isfinite(min_value) or not np.isfinite(max_value):
        raise ValueError(f"{col} の探索範囲を決められません。")
    if min_value > max_value:
        min_value, max_value = max_value, min_value

    if np.isfinite(step) and step > 0:
        count = int(math.floor((max_value - min_value) / step)) + 1
        values = min_value + np.arange(count) * step
        if values[-1] < max_value:
            values = np.append(values, max_value)
    else:
        points = max(points, 2)
        values = np.linspace(min_value, max_value, points)
    return np.asarray(values, dtype=float)


def read_number(row: pd.Series, names: list[str], default: float) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    return float(default)


def suggest_samples(
    x: pd.DataFrame,
    y: np.ndarray,
    template_model: Pipeline,
    candidates: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    direction: int,
    acquisition: str,
    n_suggestions: int,
) -> pd.DataFrame:
    suggested_rows: list[dict[str, float]] = []
    train_x = x.copy()
    train_y = y.astype(float).copy()
    remaining = remove_existing_candidates(candidates, train_x, feature_columns)

    for rank in range(1, n_suggestions + 1):
        if remaining.empty:
            break

        model = clone(template_model).fit(train_x, direction * train_y)
        mean, std = predict_mean_std(model, remaining)
        score = acquisition_values(mean, std, np.max(direction * train_y), acquisition)
        best_pos = int(np.argmax(score))
        best_row = remaining.iloc[best_pos].copy()
        predicted_objective = direction * mean[best_pos]
        std_value = std[best_pos]

        result = best_row.to_dict()
        result["rank"] = rank
        result[f"predicted_{target_column}"] = predicted_objective
        result["std"] = std_value
        result["lower_95"] = predicted_objective - 1.96 * std_value
        result["upper_95"] = predicted_objective + 1.96 * std_value
        result["acquisition"] = score[best_pos]
        suggested_rows.append(result)

        train_x = pd.concat([train_x, best_row.to_frame().T], ignore_index=True)
        train_y = np.append(train_y, predicted_objective)
        remaining = remaining.drop(remaining.index[best_pos]).reset_index(drop=True)

    columns = ["rank", *feature_columns, f"predicted_{target_column}", "lower_95", "upper_95", "std", "acquisition"]
    return pd.DataFrame(suggested_rows)[columns]


def remove_existing_candidates(
    candidates: pd.DataFrame,
    observed_x: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rounded_candidates = candidates[feature_columns].round(12)
    rounded_observed = observed_x[feature_columns].round(12)
    existing = set(map(tuple, rounded_observed.to_numpy()))
    mask = [tuple(row) not in existing for row in rounded_candidates.to_numpy()]
    return candidates.loc[mask].reset_index(drop=True)


def predict_mean_std(model: Pipeline, candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    transformed = model[:-1].transform(candidates)
    gpr = model.named_steps["gpr"]
    mean, std = gpr.predict(transformed, return_std=True)
    return np.asarray(mean), np.maximum(np.asarray(std), 1e-12)


def acquisition_values(mean: np.ndarray, std: np.ndarray, best_y: float, acquisition: str) -> np.ndarray:
    improvement = mean - best_y
    z = improvement / std
    cdf = normal_cdf(z)
    pdf = normal_pdf(z)
    if acquisition == "PI":
        return cdf
    if acquisition == "MI":
        return mean + 1.96 * std
    return improvement * cdf + std * pdf


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def display_model_results(data: pd.DataFrame, bundle: ModelBundle) -> None:
    y = data.dropna(subset=[bundle.target_column])[bundle.target_column].to_numpy(dtype=float)
    st.caption(f"選択されたカーネル: `{bundle.selected_kernel}`")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("トレーニングデータの予測結果")
        st.plotly_chart(yy_plot(y, bundle.train_prediction), use_container_width=True)
        write_metrics(bundle.train_metrics, "training data")
    with col2:
        st.subheader("クロスバリデーションによる予測結果")
        st.plotly_chart(yy_plot(y, bundle.cv_prediction), use_container_width=True)
        write_metrics(bundle.cv_metrics, "cross-validation")


def yy_plot(actual: np.ndarray, predicted: np.ndarray) -> go.Figure:
    min_value = float(min(np.min(actual), np.min(predicted)))
    max_value = float(max(np.max(actual), np.max(predicted)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers", name="prediction"))
    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="ideal",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted", height=420)
    return fig


def write_metrics(metric_values: dict[str, float], label: str) -> None:
    st.write(f"r2 for {label}: {metric_values['r2']:.3f}")
    st.write(f"RMSE for {label}: {metric_values['rmse']:.3f}")
    st.write(f"MAE for {label}: {metric_values['mae']:.3f}")


def shap_explain(bundle: ModelBundle, data: pd.DataFrame) -> str | None:
    try:
        import shap
    except ImportError:
        return None

    x = data[bundle.feature_columns].copy()
    background = x.sample(min(len(x), 50), random_state=RANDOM_STATE)
    explain_x = x.sample(min(len(x), 100), random_state=RANDOM_STATE)

    def predict_from_array(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=bundle.feature_columns)
        return bundle.model.predict(frame)

    explainer = shap.KernelExplainer(predict_from_array, background.to_numpy())
    shap_values = explainer.shap_values(explain_x.to_numpy(), silent=True, nsamples=100)
    plt.figure()
    shap.summary_plot(shap_values, explain_x, feature_names=bundle.feature_columns, show=False)
    plot_path = "work/shap_summary_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()
    return plot_path


def prediction_tab(bundle: ModelBundle) -> None:
    input_format = st.radio("予測データ形式", ["Excel", "CSV", "TSV"], horizontal=True, key="prediction_format")
    uploaded_file = st.file_uploader(
        "予測したいデータを読み込んで下さい。",
        type=file_types_for_format(input_format),
        key="test_data",
    )
    if uploaded_file is None:
        st.info("説明変数列を含むExcel、CSV、TSVをアップロードしてください。")
        return

    try:
        pred_data = read_prediction_input(uploaded_file, input_format)
    except Exception as exc:
        st.error(f"予測データの読み込みに失敗しました: {exc}")
        return

    missing = [col for col in bundle.feature_columns if col not in pred_data.columns]
    if missing:
        st.error(f"予測データに必要な列がありません: {', '.join(missing)}")
        return

    mean, std = predict_mean_std(bundle.model, pred_data[bundle.feature_columns])
    result = pred_data.copy()
    result[f"predicted_{bundle.target_column}"] = mean
    result["lower_95"] = mean - 1.96 * std
    result["upper_95"] = mean + 1.96 * std
    st.dataframe(result, use_container_width=True)
    download_dataframe(result, "予測結果を保存", "prediction")


def read_prediction_input(uploaded_file, input_format: str) -> pd.DataFrame:
    if input_format == "Excel":
        return pd.read_excel(uploaded_file, sheet_name=0)
    return read_table(uploaded_file, input_format)


def download_dataframe(data: pd.DataFrame, label: str, base_file_name: str) -> None:
    output_format = st.selectbox(
        "保存形式",
        ["CSV", "TSV", "Excel"],
        key=f"{base_file_name}_download_format",
    )
    payload, file_name, mime = dataframe_to_download(data, output_format, base_file_name)
    st.download_button(label, data=payload, file_name=file_name, mime=mime)


def dataframe_to_download(data: pd.DataFrame, output_format: str, base_file_name: str) -> tuple[bytes, str, str]:
    if output_format == "Excel":
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            data.to_excel(writer, index=False, sheet_name="results")
        return (
            buffer.getvalue(),
            f"{base_file_name}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if output_format == "TSV":
        return (
            data.to_csv(index=False, sep="\t").encode("utf-8-sig"),
            f"{base_file_name}.tsv",
            "text/tab-separated-values",
        )
    return data.to_csv(index=False).encode("utf-8-sig"), f"{base_file_name}.csv", "text/csv"


def optimization_cache_key(*parts) -> str:
    serialized = []
    for part in parts:
        if isinstance(part, pd.DataFrame):
            serialized.append(str(pd.util.hash_pandas_object(part, index=True).sum()))
        else:
            serialized.append(str(part))
    return "|".join(serialized)


if __name__ == "__main__":
    main()
