from catrobot.narration import width, wrap


def test_short_subtitle_stays_on_one_line():
    assert wrap("結果です。") == ["結果です。"]


def test_long_subtitle_splits_at_a_comma_not_inside_a_word():
    text = "ロボット用のVLA、SmolVLAに、映像を見て補正を出す先生役を学ばせ、ESP32の小さなネットワークへ蒸留します。"
    lines = wrap(text)
    assert "".join(lines) == text
    assert all(width(line) <= 40 for line in lines)
    assert not any(line.startswith("SP32") for line in lines)


def test_ascii_counts_as_half_width():
    text = "今回は東大松尾研 Physical AI 講座 応用編2026の最終課題で提出した、「まっすぐ走る学習をさせたロボ」を紹介します。"
    assert wrap(text) == [
        "今回は東大松尾研 Physical AI 講座 応用編2026の最終課題で提出した、",
        "「まっすぐ走る学習をさせたロボ」を紹介します。",
    ]
