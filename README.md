# Bayesian Optimization Streamlit App

Qiita記事「[ベイズ最適化（実験点提案）アプリをStreamlitで構築するぜ！](https://qiita.com/MAsa_min/items/3c14773aa587cafd9e94)」の断片コードをもとに、欠けている処理を補った実行可能なStreamlitアプリです。

## 起動

```powershell
pip install -r requirements.txt
streamlit run app.py
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

## 記事内容に対する変更

元の動作を保ちながら、実際に使うときに困りやすい部分を補強しています。

主な変更点は以下です。

- 入力データのチェックを追加
  - 実験データの重複列名を警告
  - 欠損値がある場合に警告
  - 探索範囲に `min` と `max` が不足している場合に警告
- 計算中の進捗表示を追加
  - 入力整理
  - カーネル候補の比較
  - 候補点生成
  - 提案点選択
- SHAP解析の設定を追加
  - SHAPに使う最大行数を画面で指定可能
  - 背景データの最大行数を画面で指定可能
  - 大きいデータで計算が重くなりすぎるのを避けやすくした
- 提案点の可視化を追加
  - 実験済みデータと提案点を同じ散布図に表示
  - X軸、Y軸、色を画面で選択可能
- カテゴリ変数に対応
  - 数値変数は中央値補完と標準化
  - カテゴリ変数は最頻値補完とOne-Hot Encoding
  - 未知カテゴリはエラーにせず無視
- 探索範囲ファイルでカテゴリ候補を指定可能
  - `values` 列に `A,B,C` のようにカンマ区切りで指定
  - `categories`, `choices`, `候補`, `カテゴリ` という列名も利用可能

## 注意

このコードは openai codex + gpt-5.5 にて作成された。
