# Bayesian Optimization Streamlit App

Qiita記事「[ベイズ最適化（実験点提案）アプリをStreamlitで構築するぜ！](https://qiita.com/MAsa_min/items/3c14773aa587cafd9e94)」の断片コードをもとに、欠けている処理を補った実行可能なStreamlitアプリです。

## 起動

```powershell
pip install -r requirements.txt
streamlit run app.py
```

改善版を試す場合:

```powershell
streamlit run app2.py
```

## 入力データ

Excel、CSV、TSVに対応しています。

Excelの場合は1ファイル2シート構成です。

- `Sheet1`: 実験済みデータ。目的変数と説明変数を列に持つ表。
- `Sheet2`: 探索範囲。indexに説明変数名、列に `min`, `max`, `step` または `points` を指定。

CSV/TSVの場合は2ファイルをアップロードします。

- 実験データ: 目的変数と説明変数を列に持つ表。
- 探索範囲: 1列目に説明変数名、列に `min`, `max`, `step` または `points` を指定。

`Sheet2` の例:

|      | min | max | step |
| ---- | --- | --- | ---- |
| x1   | 0   | 10  | 0.5  |
| x2   | 20  | 80  | 5    |

`step` が空の場合は `points`、それも空の場合は自動で候補点を作ります。組み合わせが多すぎる場合はランダムサンプリングに切り替えます。

カテゴリ変数を使う場合は、探索範囲ファイルに `values` 列を追加し、候補をカンマ区切りで指定します。

|          | min | max | step | values |
| -------- | --- | --- | ---- | ------ |
| x1       | 0   | 10  | 0.5  |        |
| material |     |     |      | A,B,C  |

## サンプルデータ

サンプルを同梱しています。

- `sample_data.xlsx`
- `sample_data.csv` と `sample_limits.csv`
- `sample_data.tsv` と `sample_limits.tsv`
- `sample_mixed.xlsx`
- `sample_mixed_data.csv` と `sample_mixed_limits.csv`

まずはこれらをアップロードすると動作を確認できます。

## 注意

このコードは openai codex + gpt-5.5 にて作成された。
