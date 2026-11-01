# 走行データから直進の補正を学ぶ自作 2 輪ロボ

猫と遊ぶロボットを目指して製作した自作の 2 輪車台について、走行ログと固定カメラで測った軌跡から直進の補正を学ぶ。PAI 応用編の最終課題に使ったコードとレポートをまとめた。

車輪速度の制御と安全停止は ESP32-S3 が行う。今回の実機評価では、補正率を Mac が ESP32 と共通の C++ コードで計算して送った。ESP32 内の補正モデルによる走行は未実施である。カメラは測定と試行間の位置戻しに使い、走行中の補正には使わない。基盤モデル SmolVLA は学習時の教師として試したが、小型モデルの直接学習を上回らなかった。

結果と工夫は [レポート](projects/cat-robot-learning/report/final-report.md)（[PDF](projects/cat-robot-learning/report/final-report.pdf)）にまとめた。

## 構成

| パス | 内容 |
| --- | --- |
| `projects/cat-robot-wiring-visual/sketches/cat_robot/` | ESP32-S3 のファーム。車輪速度の PI 制御、補正率の推論（`learned_model.h`、`student_model.h`）、停止条件、50 Hz のログ、HTTP API |
| `projects/cat-robot-wiring-visual/tests/` | ファームの C++ 制御コードのテストと、操作ページの模擬通信 |
| `projects/cat-robot-wiring-visual/diagrams/` | 配線図 |
| `projects/cat-robot-learning/src/catrobot/` | Mac 側の CLI。実機の記録と自動収集、カメラの校正とマーカー検出、時計の対応付け、学習、評価、MuJoCo のシミュレーター |
| `projects/cat-robot-learning/foundation/` | SmolVLA 教師の学習用データ変換と推論 |
| `projects/cat-robot-learning/native/` | ファームと同じ C++ を Python から呼ぶバインディング |
| `projects/cat-robot-learning/results/` | 実機の測定から得た集計、モデル、校正値、学習試行ごとの指標 |
| `projects/cat-robot-learning/report/` | レポートと図 |

撮影した動画、走行ごとの生ログ、学習済みの SmolVLA の重み、提出動画の制作台本と音声素材は含めていない。

## 動かす

Python は [uv](https://docs.astral.sh/uv/) で管理する。

```sh
cd projects/cat-robot-learning
uv sync --locked
uv run pytest -q
sh ../cat-robot-wiring-visual/tests/run-control-tests.sh
uv run catrobot simulate examples/cad-visual-config.json artifacts/first-drive --mode speed --speed 0.10 --video
```

シミュレーターは仮定した物理条件で動かす確認用であり、実機の結果ではない。詳しい手順は [projects/cat-robot-learning/README.md](projects/cat-robot-learning/README.md) にある。

## ライセンス

本リポジトリのコードと付属資料は [MIT License](LICENSE) で公開する。外部ライブラリやモデルの利用には、それぞれのライセンスが適用される。
