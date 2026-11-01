# 猫じゃらしロボの直進学習

自作ロボットの出力・左右の回転と、スマホで測った軌跡を対応させ、直進に必要な補正を学ぶ。ESP32 は車輪速度の制御と安全停止を行う。回帰モデルの推論も ESP32 と Mac で共通の C++ コードに実装したが、今回の実機評価では Mac で補正率を計算して送った。ESP32 内のモデルによる走行は未実施である。カメラは測定と試行間の位置戻しに使い、走行中の補正には使わない。

実機での結果と、基盤モデル SmolVLA を教師にした蒸留の結果は [レポート](report/final-report.md) にまとめた。

## まずシミュレーターの走行を見る

リポジトリのルートから実行する。Pythonはuvで管理する。

```sh
cd projects/cat-robot-learning
uv sync --locked
uv run catrobot simulate examples/cad-visual-config.json artifacts/first-drive --mode speed --speed 0.10 --video
```

最初に `artifacts/first-drive/simulation.mp4` を再生する。説明スライドを挟まず、1フレーム目から車体を映す。カメラは車体を追い、床の模様とタイヤの白いマークで移動と回転を確認できる。時刻、前進距離、目標速度も画面に表示する。`poster.png` は保存したMP4の先頭フレーム、`video-check.json` は全フレームで青い車体が一定以上の面積を占めるかを確認した記録である。表示の検査に失敗した場合は例外を返し、生成した映像と記録を残す。

この実行は仮の物理条件での動作確認であり、実機の直進性能を確認した結果ではない。走行が見えることを確認してから、テストと学習・評価の一括実行へ進む。

```sh
uv run pytest -q
uv run catrobot demo artifacts/my-simulation --config examples/cad-visual-config.json --video
```

`artifacts/my-simulation/` に、校正用3走行、学習用30走行、評価用30走行、回帰モデル、比較PNG/PDF、CSV、説明PDF、3種類の制御のMP4ができる。同じ出力先は上書きしない。これらは仮定した物理条件でのシミュレーションであり、実機の測定結果ではない。補正が目標を達成しなかった場合も結果を残す。

C++ヘッダーを編集した後は、ネイティブモジュールも作り直す。

```sh
uv sync --reinstall-package catrobot-learning
sh ../cat-robot-wiring-visual/tests/run-control-tests.sh
sh scripts/build-firmware.sh --build-path /tmp/cat-robot-learning-firmware-build
```

ビルドだけでは実機へ書き込まれない。Apple Siliconでは公式ctags 5.8-arduino11を利用者のキャッシュでビルドし、Arduino15を変更せずに使用する。

COM側のUSBシリアル変換器を使う場合は `sh scripts/build-firmware.sh --com` でビルドする。標準は基板のUSB側にシリアル入出力を向け、`--com` はUART0を使う。2026年10月31日にCOM用の1.1.0を書き込み、起動時停止と、USBシリアルの `telemetry` による直近16件ずつ2回のログ保存を確認した。各応答内の採取間隔は20 msであった。この診断は連続収集ではなく、走行時は下記のWi-Fi収集を使う。実機の診断記録は `artifacts/firmware-com-20261031/` に保存した。

## 実装の構成

- `../cat-robot-wiring-visual/sketches/cat_robot/` が実機とネイティブモジュールの共通C++実装である。既存の停止条件、PI速度制御、回帰モデル、20 ms周期を目標とした計測バッファを持つ。
- `src/catrobot/simulation.py` はMuJoCoの車体・左右輪・支持球を作り、共通C++が出した8 bitのPWM値でモーターを駆動する。速度を直接固定する理想的な車輪には置き換えない。
- `acquisition.py` は実機の記録、`vision.py` は動画の校正とマーカー検出、`sync.py` はLEDによる時計の対応付け、`model.py` は学習、`report.py` は独立評価の集計を行う。
- モーターの物理応答と、制御に使うfeedforwardの校正値を別々に扱う。シミュレーターのトルクは計算値であり、実機のトルク・電流は取得していない。

Shapr3Dの筐体STEPから変換した3部品のCADメッシュも同梱する。設計外形は150 × 150 × 49 mmである。`examples/cad-visual-config.json` はCADの+Yを前方とし、左右のゴムタイヤ2個と、前後の凹みのZ−側に置くボール車輪2個を設定する。球径・取り付け高さ・質量・接触は未校正として区別する。取り込み条件と再現コマンドは [CAD.md](CAD.md) に記載する。

## 実機の校正

### ステップ1: 回転カウントと形状

新しいファームウェアの書き込みと起動時停止を確認してから進める。既存のUSBと外部5 Vの分離は未確認なので、USB確認時は走行用電池を外す。

```sh
mkdir -p data
cp -n examples/cad-visual-config.json data/geometry.json
uv run catrobot record-serial data/calibration/turns-1.jsonl --stop-file data/calibration/stop
```

既存の `data/geometry.json` は上書きしない。既に校正値がある場合はその値を保存し、CADの向きと前後2個の `ball_casters` を反映して使う。`record-serial` はCOM側のUSBシリアル（VID 1A86、PID 55D3）で状態を読むだけで、操作を有効化しない。有効化中や出力が0でない状態を検出すると記録を止める。記録中、左右の車輪をそれぞれ車体の前進方向へ10回転させ、`--stop-file` に指定したファイルを作ると終了する。同じ記録を2回取り、続けて床のテープに沿って車体を手で1.00 m押す記録も2回取る。

```sh
uv run catrobot wheel-calibration --turn-log data/calibration/turns-1.jsonl --turn-log data/calibration/turns-2.jsonl --roll-log data/calibration/roll-1.jsonl --roll-log data/calibration/roll-2.jsonl
```

1回転あたりのカウントと符号は10回転から、実効の車輪半径は1 m押しから求める。2回の差が1%を超える場合と、不正な遷移がある場合は `accepted` をfalseにする。手で押した軌跡は完全な直線ではないため、左右の半径差は参考値として表示し、共通の半径を使う。車輪の実際の周長、左右輪間隔、車体寸法、マーカーの床からの高さと車体基準点からの位置も測る。値を `data/geometry.json` に記録し、確認できた時点で `provenance` を `measured` にする。初期値は例であり、手持ちの実物の仕様ではない。

`left` と `right` の `radius_m`、`counts_per_revolution`、`encoder_sign` が距離換算に使われる。`feedforward` は速度1 m/sあたりの出力割合、`deadband_percent` は始動に必要な出力割合であり、次のステップで測る。

### ステップ2: 同期LEDとモーター応答

まず走行せずに記録を予行する。`record` は停止中のWi-Fi記録、出力0の `collect` はモーターを回さずに有効化・指令・記録・同期LEDを通しで動かす。予行は `data/pwm/` と別の場所へ保存する。出力0の試行は `fit-motors` の対象から外す。

```sh
uv run catrobot record data/rehearsal/r1.jsonl --duration 60
uv run catrobot collect data/geometry.json data/rehearsal/r2-001 --mode pwm --pwm 0 --split rehearsal
```

各試行の終わりにTerminalへ1行の品質要約を出す。ESP32の採取間隔、抜けた連番、ログ回収の往復時間、指令の間隔を別々に示す。詳細は試行ごとの `quality.json`、通信の記録は `transport.jsonl` に残る。Macが受け取った時刻は、ESP32が採取した時刻として使わない。

GPIO8、220 Ωまたは330 Ω、赤LEDのアノードを直列につなぎ、カソードをGNDへつなぐ。LEDはマーカーと同じ高さに置く。初期の位置は、マーカー左上を(0, 0)、右下を(1, 1)としたときの(1.2, 0.15)である。実際の位置を `led_offset` に記録する。

車輪を浮かせて回転方向と停止を確認してから、猫がいない平らな走行領域で行う。USBを外した状態で電池を使う。スマホでLEDの点滅から停止後の点滅まで撮る。次のコマンドは実際にモーターを動かす。

```sh
uv run catrobot collect data/geometry.json data/pwm/pwmcal-001 --mode pwm --pwm 20 --speed 0.05
uv run catrobot collect data/geometry.json data/pwm/pwmcal-002 --mode pwm --pwm 30 --speed 0.05
uv run catrobot collect data/geometry.json data/pwm/pwmcal-003 --mode pwm --pwm 40 --speed 0.05
uv run catrobot fit-motors data/geometry.json data/pwm data/measured-config.json
```

収集は1.3 m相当の回転または最大25秒で止める。手動停止、通信断、encoder無応答でも停止する。実験中のサーボ信号は0である。`fit-motors` は定常区間からfeedforwardを求め、始動区間から実効的な一次遅れも推定する。これは実機の軸トルクを測ったものではなく、接触の物理モデル全体を検証したものでもない。推定誤差は `response_fit_rms_m` に保存し、`physics_calibrated` は自動でtrueにしない。

## スマホ動画の校正

撮影は固定した同じレンズ・解像度・ズームで行い、手ぶれ補正や途中の画角変更を避ける。30 fps以上を目安にし、実際のフレーム時刻を解析する。

```sh
uv run catrobot markers artifacts/print
```

SVGを100%で印刷し、マーカーの黒い正方形が50 mm、チェッカーボードの一辺が25 mmであることを定規で確認する。ボードを異なる位置と向きで写した動画を `data/camera-calibration.mp4` に置く。

```sh
uv run catrobot calibrate-camera data/camera-calibration.mp4 data/camera.json
```

床の座標はXが走行方向、Yがその左、Zが上である。CADでは+Yが車体の前方であり、床や制御のX方向と対応する。床にID 1〜4を長方形の角へ貼る。1が出発側の左、2が奥の左、3が奥の右、4が出発側の右で、どのマーカーも上辺を左（+Y）、右辺を走行方向（+X）へ向ける。各マーカーの黒い正方形の同じ角どうしの距離を、4辺と対角線2本について巻き尺で測る。

```sh
uv run catrobot arena-layout data/arena-layout.json --d12 1.600 --d23 0.800 --d34 1.600 --d41 0.800 --d13 1.789 --d24 1.789
```

6つの距離から各マーカーの角の座標を最小二乗で求める。距離の残差が5 mmを超える場合は `accepted` をfalseにする。ID 0を車体に貼り、マーカーの右辺へ向かう方向を車体の前方（CADの+Y）にする。向きを変える場合は `marker_yaw_deg` を設定する。

```sh
uv run catrobot frame data/train-video.mp4 data/arena-frame.png
uv run catrobot calibrate-arena data/arena-frame.png data/camera.json data/arena-layout.json data/arena.json
uv run catrobot track data/train-video.mp4 data/camera.json data/arena.json data/measured-config.json data/train-track.jsonl
```

床に1.00 m離して貼った2本のテープの位置へ車体を5秒ずつ置いて撮り、動画から求めた距離を確かめる。

```sh
uv run catrobot track data/distance-check.mp4 data/camera.json data/arena.json data/measured-config.json data/distance-track.jsonl
uv run catrobot check-distance data/distance-track.jsonl --expected-m 1.0
```

小さなマーカーの奥行き推定だけに依存せず、カメラの光線と実測したマーカー高さの平面を交差させて位置を求める。平らな床と一定のマーカー高さを前提とし、床の平面へ直接投影することによる高さの誤差を避ける。床上の既知距離でも精度を確認する。

## 学習用30走行と評価用30走行

`schedule` は校正した最大速度の45%、80%を初期の2速度として選ぶ。必要なら `--low` と `--high` で指定する。補正率は左右の目標速度を `v×(1−補正率)`、`v×(1＋補正率)` にする値である。

1 m進む間の向きの変化は約 2×補正率÷左右輪間隔である。間隔0.14 mでは補正率0.04で約33°曲がり、レーンの外へ出うる。そこで、まず補正0で各速度2回の試走（`--split probe`）を行い、曲がり方の偏りを打ち消す補正率を中心にして、刻みの−2〜+2倍の5段階、各3回の同定を行う。下の `--center` と `--step` は例であり、試走の結果から決める。

```sh
uv run catrobot schedule data/probe-schedule.json --config data/measured-config.json --split probe
uv run catrobot collect-schedule data/measured-config.json data/probe-schedule.json data/probe
uv run catrobot schedule data/train-schedule.json --config data/measured-config.json --split train --center -0.01 --step 0.01
uv run catrobot collect-schedule data/measured-config.json data/train-schedule.json data/train --session-id train-1
```

1試行ごとに車体の位置と向きを戻して確認に答える。最初の点滅前から最後の点滅後まで、1つのセッションを同じ動画に記録する。途中で失敗した試行は `failed/<試行ID>-a<番号>/` へ移して残し、同じ試行IDで撮り直せる。試みの一覧、起動識別子、コードのコミットは `session-<ID>.json` と各試行の `metadata.json` に残る。危ないときはTerminalでCtrl-Cを押す。停止指令を送り、終了の点滅を出して、介入として記録する。

```sh
uv run catrobot analyze-session data/train train-1 data/train-track.jsonl
uv run catrobot validate data/train
uv run catrobot train data/train data/model.json
uv run catrobot export-model data/model.json ../cat-robot-wiring-visual/sketches/cat_robot/learned_model.h
sh scripts/build-firmware.sh --build-path /tmp/cat-robot-learning-firmware-build
```

実測モデルを書き込んで起動時停止を確認した後、評価する。実機の収集処理は、ファームウェア内のモデル識別子が指定モデルと一致することを確認する。

```sh
uv run catrobot schedule data/test-schedule.json --config data/measured-config.json --split test
uv run catrobot collect-schedule data/measured-config.json data/test-schedule.json data/test --model data/model.json --session-id test-1
uv run catrobot track data/test-video.mp4 data/camera.json data/arena.json data/measured-config.json data/test-track.jsonl
uv run catrobot analyze-session data/test test-1 data/test-track.jsonl
uv run catrobot report data/test artifacts/measured-report --model data/model.json
```

`analyze-session` は試行の順番ではなく時刻で動画と記録を対応付ける。各試行の開始・終了の点滅（点灯と消灯が計6回）を、ESP32の時刻と動画の時刻で照合する。点灯の切り替わりは前後2つの観測の中点とし、20 msの記録間隔とフレーム間隔による量子化の誤差を半分にする。点滅を見落とした試行や途中で失敗した試行があっても、他の試行はずれない。残差が1フレームを超える試行は不採用とし、理由を `metrics.json` と `sync-<ID>.json` に残す。`validate` は、学習の前に全試行の列、欠落、時刻の単調増加、停止理由、動画解析の結果を検査し、不合格の理由を `validation.json` に書く。

評価は出力指定、PI、PIと学習補正の各制御を2速度で5回ずつ行う。順序は固定シードで入れ替える。学習用と評価用のIDは分け、同じ試行や動画のフレームが両方へ入ることを拒否する。

## 無人での収集（2026年11月1日の構成）

印刷物と人手の代わりに、次の構成で収集した。手順はREADME前半の印刷マーカー・同期LEDの構成と置き換えられる。

- 床の基準は30 cm角3×3のテープのマス目である。カメラはPixel 9aをUSBのWebカメラとしてMacにつなぎ、マス目の交点16個と実測したレンズ高さ106 cmから校正する（`calibrate-grid`）。
- 車体マーカーは手描きのArUco ID 0（6 cm）である。四隅の光線をマーカー高さの平面と交差させて位置と向きを求める。マーカーが車体前方から回っている角度は `marker_yaw_deg` に入れる。
- ロボットは家の2.4 GHz Wi-Fiに参加する（ファーム1.3.1以降）。ネットワーク名とパスワードはUSBシリアルの `sta-ssid` と `sta-pass` で保存し、チャットや記録には残さない。指令は別スレッドから100 msごとに送り、300 msの指令期限は変えない。
- ボードのLEDが上から見えない場合は、エンコーダー距離とカメラ距離の時間ずれで動画とログを対応付ける（`analyze-motion`）。

```sh
uv run catrobot calibrate-grid data/calibration/webcam/frame.png data/calibration/webcam/camera.json data/calibration/webcam/arena.json --height-m 1.06
uv run catrobot autocollect data/measured-config.json data/train-schedule.json data/train data/video/live/latest.jpg data/calibration/webcam/camera.json data/calibration/webcam/arena.json train-1 --url http://192.168.11.33 --stop-m 0.9 --goal-m 0.7
uv run catrobot track data/video/live/train-1.mp4 data/calibration/webcam/camera.json data/calibration/webcam/arena.json data/measured-config.json data/train/track-train-1.jsonl
uv run catrobot analyze-motion data/train train-1 data/train/track-train-1.jsonl data/video/live/train-1.mp4.start data/measured-config.json
```

`autocollect` は、各試行の前にカメラで車体をスタート位置へ戻す。戻すときだけカメラを使い、短い出力パルスで回転と直進を繰り返し、パルスごとに止まって3枚の中央値で姿勢を確かめる。試行中の走行にはカメラを使わない。通信の問題で止まった試行（`acquisition_error`、`command_timeout`）だけを最大2回撮り直し、制御の結果で止まった試行は記録どおり残す。録画はffmpegでMP4に書き、同時に0.25秒ごとの静止画を出して姿勢の確認に使う。

評価の「学習補正」は、ファームへモデルを書き込む前でも、`--host-model` を付けるとファームと同じC++の `choose()` でMac側が補正率を選び、ESP32のPIがその補正率で走る。書き込み後のファーム内の推論との一致は別に確認する。

### ESP32に書き込んだモデルでの評価

`export-model` と `export-student` でRidgeと学生をスケッチのヘッダーへ書き出し、ファームを書き込んだ後、`device` 区分（高い方の速度で、PI、ESP32内のRidge、ESP32内の学生を各5回）を走らせる。学生モード（`mode=3`）はESP32が100 msごとに補正率を推論し、Macは目標速度と指令の期限だけを送る。`--student` に渡した学生のIDと、ファームが報告する `student_id` が一致しない場合は走らせない。

```sh
uv run catrobot schedule data/device-schedule.json --split device --config data/measured-config.json
uv run catrobot autocollect data/measured-config.json data/device-schedule.json data/device data/video/live/latest.jpg data/calibration/webcam/camera.json data/calibration/webcam/arena.json device-1 --url http://192.168.11.33 --stop-m 0.9 --goal-m 0.7 --start-x 0.12 --start-y 0.5 --model data/model.json --student data/student-direct-2.json
```

## 学習と判定の内容

Ridge回帰の入力は目標速度、左右補正率、その積、補正率の2乗である。教師データは開始時の車体の向きを基準とした1 m先での横ずれと向きの変化である。正規化と回帰の係数を保存し、0.001刻みの補正候補をC++で比較する。学習した速度・補正率の範囲を超えて推論しない。候補の損失は横ずれ5 cm、向き5度をそれぞれ1として二乗和で比較する。

目標は1 mの前進で最大横ずれ5 cm以内、向きの変化5度以内である。最大横ずれ、向き、実際の速度、到達率を併記し、未到達と解析品質不足を除いて成功率を計算しない。動画と機体の時計は開始・終了のLEDで対応付け、同期誤差が1フレームを超える場合は採用しない。明示的な同期点を使う場合は `analyze --anchors` に `[動画の秒, 機体の秒]` の組をJSONで渡す。

単純なPI制御を学習補正が上回らない場合や、補正後も目標を満たさない場合も記録する。同速の車輪が必ず直進を意味するとは扱わない。外部から押された場合の位置修正や、別の床・車体への汎化は今回の検証対象外である。

## 提出物の準備

```sh
uv run catrobot package-submission artifacts/submission-review --model data/model.json --training-runs data/train --report artifacts/measured-report
```

許可したコード、モデル、学習用の数値、評価CSVとPDFだけを別ディレクトリへ写す。生のスマホ動画、認証情報、スターターキットの教材PDF、他の工作、Git履歴は含めない。モデルがない段階は `--model` と `--training-runs` を省いてコードだけを準備できる。コマンドは公開リポジトリの作成やアップロードを行わない。

動画は `examples/video-manifest.json` の形式で、各素材の `source` を `measured`、`simulated`、`explanation` として指定する。2倍速の場合は `speed=2` として自動で表示する。`crop` は素材の切り出し範囲 `[x, y, 幅, 高さ]` である。

```sh
uv run catrobot make-video data/video-manifest.json artifacts/presentation.mp4
```

60〜180秒のMP4だけを受け付ける。日本語字幕には `--font` で日本語フォントを指定する。提出前に映像の個人情報を確認し、PDFのファイル名を講座指定のアカウントID付きに変更する。公開と提出は成果物を確認した後に行う。

## 参照とライセンスの扱い

- [MuJoCoのモデルとアクチュエーター](https://mujoco.readthedocs.io/en/stable/modeling.html)
- [OpenCVのArUcoマーカー検出](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [scikit-learnのRidge回帰](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
- [Arduino ctagsの原典](https://github.com/arduino/ctags/tree/5.8-arduino11)

図のスタイルと補助関数はローカルのpaper-figuresスキルのpaperfig v0.2.0から必要な部分をそのまま収録した。外部ライブラリの実装本体はコピーせず、uv.lockで依存バージョンを固定する。公開用のスナップショットは、公開前に内容とライセンス表記を確認する。
