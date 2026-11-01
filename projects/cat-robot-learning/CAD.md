# Shapr3Dの筐体データ

## 取り込んだ形状

2026年10月29日にユーザーが提供した `shapr3d_export_2026-10-29_09h14m.step` を使用する。STEPの単位はmmである。Open CASCADEで形状の有効性を確認し、次の3ソリッドを元の座標と名前のまま取り込んだ。全体の外形はX、Y、Zの順に150 × 150 × 49 mmである。これらはCADの設計値であり、印刷後の実測値ではない。変換結果は [manifest.json](src/catrobot/assets/shapr3d-20261029/manifest.json) に保存する。

| STEPの名前 | X方向の範囲（mm） | Y方向の範囲（mm） | Z方向の範囲（mm） |
| --- | ---: | ---: | ---: |
| ボディ 03 | 60.470〜75.000 | −25.000〜25.000 | 43.000〜49.000 |
| ボディ 02 | −75.000〜−60.470 | −25.000〜25.000 | 43.000〜49.000 |
| ボディ 01 | −70.711〜70.711 | −75.000〜75.000 | 0.000〜49.000 |

元のSTEPは `data/cad/` にそのまま保存し、SHA-256と受領日を `receipt.json` に記録した。このディレクトリはGitと提出用スナップショットの対象外である。コードに含めるSTEPでは、書き出しヘッダーのユーザー名・パスと、Shapr3Dの `KEY` プロパティに入っていた一時ファイルのパスを除いた。形状の構成要素と部品名は保持する。処理は [cad.py](src/catrobot/cad.py) の `sanitized_step` と `convert_step` に実装する。

## 変換と表示

通常のシミュレーションと表示には変換済みのOBJを使う。頂点の単位はmであり、MuJoCoでさらに0.001倍しない。穴と切り欠きを含め、3部品を別々のメッシュにする。変換時の線形偏差は0.15 mm、角度偏差は0.15 radである。出力ファイルのハッシュ、体積、均一な密度を仮定した幾何学的な中心は `manifest.json` に記録する。印刷時の充填率、電池・モーターなどの重量を反映した機体の重心は求めていない。

プロジェクトのディレクトリで実行する。

```sh
uv run catrobot view-cad artifacts/chassis.html --config examples/cad-visual-config.json
uv run catrobot simulate examples/cad-visual-config.json artifacts/cad-preview-trial --video
```

`chassis.html` は外部ライブラリやサーバーを必要とせず、ドラッグによる回転、拡大、座標面ごとの表示、各部品の表示切り替えを行える。`cad-preview-trial/` には走行記録と動画に加え、`robot.xml` と使用したメッシュを保存する。XMLはそのディレクトリ内のファイルだけで再読込できる。[cad_view.html](src/catrobot/cad_view.html) と [simulation.py](src/catrobot/simulation.py) が実装である。

CADを表示した状態で、モーターの模擬校正、30走行の学習、30走行の評価、比較図・PDF・動画まで実行する場合は、設定ファイルを `demo --config` へ渡す。指定した物理パラメーターを使うため、既定のデモの左右差で上書きしない。結果は引き続き仮定した物理条件でのシミュレーションである。[cli.py](src/catrobot/cli.py) が処理を呼び出す。

```sh
uv run catrobot demo artifacts/cad-learning-preview --config examples/cad-visual-config.json --video
```

STEPから変換し直す場合だけ、Python 3.12〜3.14と追加の `cad` 依存グループを使う。Open CASCADEのPythonバインディングは変換用に固定し、通常のシミュレーションでは読み込まない。[pyproject.toml](pyproject.toml) と [uv.lock](uv.lock) に依存条件と解決したバージョンを記録する。

```sh
uv sync --locked --python 3.12 --group cad
uv run --group cad catrobot import-cad src/catrobot/assets/shapr3d-20261029/chassis.step artifacts/cad-rebuilt
uv run --group cad pytest -q
```

変換先と走行の出力先は既存のディレクトリを上書きしない。別のSTEPを調べる場合は `import-cad` の出力を `view-cad --bundle` へ渡せる。

```sh
uv run catrobot view-cad artifacts/rebuilt-chassis.html --bundle artifacts/cad-rebuilt
```

## 走行シミュレーションでの扱い

2026年10月29日、ユーザーがCADの+Yを前方と確認した。続いて、ゴムタイヤ2個はX軸側、ボール車輪2個はY軸側にある凹みのZ−側へ装着していると説明した。これに基づき、車体の方向を次のように対応づける。

| 車体の方向 | CADの方向 | シミュレーターの方向 |
| --- | --- | --- |
| 前 | +Y | +X |
| 左 | −X | +Y |
| 右 | +X | −Y |
| 上 | +Z | +Z |

`examples/cad-visual-config.json` はZ軸回りに−90°回すクォータニオン `[√0.5, 0, 0, −√0.5]` を設定する。`alignment=orientation_confirmed` は向きの対応を確認済みという意味であり、取り付け位置や高さを実測済みという意味ではない。外形の中心を車体基準点に置くZ方向−24.5 mmの移動は仮のままである。`cad_visual.position_m` は回転後のCAD原点の位置、`quaternion_wxyz` はその回転である。[examples/cad-visual-config.json](examples/cad-visual-config.json) に設定を保存する。

`view-cad --config` は元のCAD座標を保って表示し、前・左・上の対応を設定から表示する。`--config` を渡さない形状だけの表示では向きを確定しない。元のSTEPと形状変換のマニフェストには、実機の組立情報を埋め込まない。

CADメッシュは質量0、接触を無効にした表示専用の形状として扱う。質量・慣性と床への接触には、既存の箱・車輪・支持球による仮の物理モデルを使う。MuJoCoの通常のメッシュ衝突は凸形状として計算されるため、筐体の穴や切り欠きが自動的に接触形状へ反映されるとは扱わない。これは表示を追加する際の実装判断であり、実機との接触を検証した結果ではない。[MuJoCo, 2026/10, Computation: Collision detection](https://mujoco.readthedocs.io/en/stable/computation/#collision-detection)

車輪、モーター、支持輪を独立した部品として識別できる情報や、関節・可動軸の指定はこの取り込みで得ていない。取り付け位置、車輪径、支持輪の詳細寸法と位置、床からの高さ、機体全体の重量、モーター応答、摩擦は確認・実測が必要である。CADだけから取り付け不良や曲がる原因を断定しない。

## タミヤのボール車輪

2026年10月29日、ユーザーからタミヤのボール車輪を2個、Y軸側の凹みのZ−側へ前後に装着していると報告された。発注記録の秋月110372は、タミヤのボールキャスター70144に対応する。装着数2個の根拠はユーザー報告であり、販売単位の2セット入りから推定した数ではない。実物の型番表記、球径、組立高さは未確認である。[秋月, 2026/10, 型番](https://akizukidenshi.com/catalog/g/g110372/)

メーカーは金属球を3本のステンレス軸で支える構造、完成時の取り付け全幅20 mm・全高11〜35 mmと説明している。この全高を球の直径として使わず、実機の球径と筐体からの突出量を別々に確認する。[タミヤ, 2026/10, 基本スペック](https://www.tamiya.com/japan/products/70144/index.html)

STEPの水平面と穴の位置を確認すると、2つの凹みはXY方向に30 × 30 mm、中心はCADの（0, +50）mmと（0, −50）mmである。対応する水平面のZ座標は22 mmと33 mmであった。これらの値を、球の取り付け高さや床の位置として確定しない。元の [chassis.step](src/catrobot/assets/shapr3d-20261029/chassis.step) が形状の原典である。

設定ファイルの `ball_casters` に2個の球を記録し、球面関節で受動的に転がす。最初のモデルでは凹み中心の真下へ置き、車体基準点に対する座標を前（+50, 0, −41.5）mm、後（−50, 0, −41.5）mmとする。半径10 mm・質量10 gは仮定値であり、タミヤ製品の実測値ではない。Z位置は、仮のゴムタイヤの接触面 `−30 − 21.5 = −51.5 mm` に球の下面を合わせた値である。これは4輪が同じ床面に接する基準条件であり、実機の取り付け高さが一致したという結果ではない。支持金具の形状と質量も再現していない。[simulation.py](src/catrobot/simulation.py) と [config.py](src/catrobot/config.py) が実装である。

古い実験の再現のため、`ball_casters` のない設定は従来の後ろ1個の仮モデルとして読む。実機に対応する配置の開始点には `examples/cad-visual-config.json` を使う。球の配置・向きが変わった構成は別の識別子となり、旧構成で学習したモデルを使い回さない。

## 検証と記録

[test_cad.py](tests/test_cad.py) で、mmからmへの換算、3部品の位置と外形、STEPの再変換による同一メッシュの生成、個人パスの除去、公開用スナップショットの内容を確認する。CAD表示の有無で質量・慣性と同じ入力に対する走行軌跡が一致することも確認する。これにより表示の追加で既存の物理条件を変更していないことを確認する。

走行記録にはCADのハッシュ、仮の配置、`collision_model=primitive_proxy` を残す。配置を変更した構成は別の構成ハッシュとなり、異なる構成で学習したモデルを適用しない。CADを使わない既存の構成のハッシュは保持する。これらの実装と確認条件は [config.py](src/catrobot/config.py)、[simulation.py](src/catrobot/simulation.py)、[test_cad.py](tests/test_cad.py) に対応する。

## 参照

[MuJoCo, 2026/10] Google DeepMind. "Computation: Collision detection." MuJoCo Documentation. <https://mujoco.readthedocs.io/en/stable/computation/#collision-detection>

[秋月, 2026/10] 秋月電子通商. "ボールキャスター（2セット入り）110372." <https://akizukidenshi.com/catalog/g/g110372/>（2026年10月29日確認）

[タミヤ, 2026/10] 株式会社タミヤ. "ボールキャスター（2セット入）70144." <https://www.tamiya.com/japan/products/70144/index.html>（2026年10月29日確認）
