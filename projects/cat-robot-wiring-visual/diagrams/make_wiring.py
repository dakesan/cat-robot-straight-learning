"""Generate a Wokwi-style wiring picture of the cat robot (SVG + PNG).

Usage: python3 make_wiring.py
Writes wiring.svg next to this script and renders wiring.png with headless Chrome.
"""
import pathlib
import subprocess

W, H = 1980, 1250
OUT = pathlib.Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONT = "Hiragino Sans, Helvetica, Arial, sans-serif"

parts = []   # component drawings (under wires)
wires = []   # wires
tops = []    # pin dots, junctions, labels (over wires)

COL = {
    "pos": "#e0262b", "gnd": "#222222", "v33": "#f39c12", "v5": "#c0392b",
    "v6": "#8e44ad", "pwm1": "#2e86de", "pwm2": "#16a085", "pwm3": "#27ae60",
    "pwm4": "#0abde3", "enc1": "#d35400", "enc2": "#b8860b", "enc3": "#6c5ce7",
    "enc4": "#e84393", "servo": "#f1c40f", "adc": "#00a8a8", "mot": "#7f8c8d",
}


def text(x, y, s, size=13, fill="#222", anchor="start", weight="normal", layer=None):
    (layer if layer is not None else tops).append(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def pin(x, y):
    tops.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="#d9d9d9" stroke="#555" stroke-width="1.5"/>')


def dot(x, y, color):
    tops.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" stroke="#fff" stroke-width="1.5"/>')


def wire(pts, color, dashed=False, r=10):
    """Orthogonal polyline with rounded corners."""
    d = f"M{pts[0][0]},{pts[0][1]}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        l1 = max(abs(x1 - x0), abs(y1 - y0))
        l2 = max(abs(x2 - x1), abs(y2 - y1))
        rr = min(r, l1 / 2, l2 / 2)
        sx = (x1 - x0) / l1 if l1 else 0
        sy = (y1 - y0) / l1 if l1 else 0
        ex = (x2 - x1) / l2 if l2 else 0
        ey = (y2 - y1) / l2 if l2 else 0
        d += f" L{x1 - sx * rr},{y1 - sy * rr} Q{x1},{y1} {x1 + ex * rr},{y1 + ey * rr}"
    d += f" L{pts[-1][0]},{pts[-1][1]}"
    dash = ' stroke-dasharray="10 7"' if dashed else ""
    wires.append(f'<path d="{d}" fill="none" stroke="#00000033" stroke-width="7.5" '
                 f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')
    wires.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="5" '
                 f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')


def rect(x, y, w, h, fill, stroke="#333", rx=6, extra=""):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="2" {extra}/>')


# ---------------------------------------------------------------- components
def battery():
    rect(40, 450, 150, 230, "url(#lipo)", rx=10)
    rect(55, 500, 120, 120, "#1f3a93", stroke="#0f1f55", rx=4)
    text(115, 540, "2S LiPo", 18, "#fff", "middle", "bold", parts)
    text(115, 565, "7.4 V", 15, "#fff", "middle", layer=parts)
    text(115, 588, "1200 mAh", 15, "#fff", "middle", layer=parts)
    text(180, 515, "+", 18, COL["pos"], "end", "bold")
    text(180, 640, "−", 20, "#000", "end", "bold")
    pin(190, 520)
    pin(190, 620)


def fuse(x, y):
    rect(x, y - 16, 100, 32, "#3d3d3d", rx=8)
    rect(x + 22, y - 9, 56, 18, "#dfe6e9", stroke="#999", rx=8)
    parts.append(f'<line x1="{x + 24}" y1="{y}" x2="{x + 76}" y2="{y}" stroke="#7f8c8d" stroke-width="2"/>')
    text(x + 50, y - 26, "ヒューズ 5 A", 13, anchor="middle", weight="bold")
    text(x + 50, y + 36, "溶断・交換待ち", 12, "#c0392b", "middle")
    pin(x, y)
    pin(x + 100, y)


def toggle(x, y, label):
    rect(x, y - 22, 80, 44, "#555", rx=6)
    parts.append(f'<circle cx="{x + 40}" cy="{y}" r="12" fill="#bdc3c7" stroke="#333"/>')
    parts.append(f'<line x1="{x + 40}" y1="{y}" x2="{x + 52}" y2="{y - 34}" stroke="#e74c3c" '
                 f'stroke-width="8" stroke-linecap="round"/>')
    text(x + 40, y - 46, label, 13, anchor="middle", weight="bold")
    text(x + 8, y + 16, "IN", 10, "#fff", layer=parts)
    text(x + 72, y + 16, "OUT", 10, "#fff", "end", layer=parts)
    text(x + 46, y + 44, "LED GND", 10, "#555")
    pin(x, y)
    pin(x + 80, y)
    pin(x + 40, y + 40)


def buck(x, y, title, volt, color):
    rect(x, y, 200, 120, "#1e5aa8", stroke="#0d2f5c")
    rect(x + 70, y + 30, 56, 56, "#4a4a4a", stroke="#222", rx=4)
    parts.append(f'<circle cx="{x + 98}" cy="{y + 58}" r="18" fill="#2d2d2d"/>')
    rect(x + 145, y + 18, 34, 28, "#2980b9", stroke="#1a5276", rx=3)
    parts.append(f'<circle cx="{x + 162}" cy="{y + 32}" r="6" fill="#f1c40f" stroke="#7d6608"/>')
    text(x + 100, y - 12, title, 15, anchor="middle", weight="bold")
    parts.append(f'<rect x="{x + 135}" y="{y + 92}" width="60" height="22" rx="4" fill="{color}"/>')
    text(x + 165, y + 108, volt, 13, "#fff", "middle", "bold", parts)
    text(x + 8, y + 44, "IN+", 11, "#fff", layer=parts)
    text(x + 8, y + 84, "IN−", 11, "#fff", layer=parts)
    text(x + 192, y + 44, "OUT+", 11, "#fff", "end", layer=parts)
    text(x + 192, y + 84, "OUT−", 11, "#fff", "end", layer=parts)
    text(x + 8, y + 112, "MP1584EN", 10, "#cfe2ff", layer=parts)
    for px, py in ((x, y + 40), (x, y + 80), (x + 200, y + 40), (x + 200, y + 80)):
        pin(px, py)


ESP_L = ["3V3", "3V3", "RST", "4", "5", "6", "7", "15", "16", "17", "18", "8", "3",
         "46", "9", "10", "11", "12", "13", "14", "5V", "GND"]
ESP_R = ["GND", "TX", "RX", "1", "2", "42", "41", "40", "39", "38", "37", "36", "35",
         "0", "45", "48", "47", "21", "20", "19", "GND", "GND"]
ESP_X0, ESP_X1, ESP_Y0, ESP_DY = 1500, 1700, 110, 27


def esp_l(name, nth=0):
    idx = [i for i, n in enumerate(ESP_L) if n == name][nth]
    return (ESP_X0, ESP_Y0 + idx * ESP_DY)


def esp_r(name, nth=0):
    idx = [i for i, n in enumerate(ESP_R) if n == name][nth]
    return (ESP_X1, ESP_Y0 + idx * ESP_DY)


def esp32(used_l, used_r):
    rect(ESP_X0, 70, 200, 660, "#1b1b1b", stroke="#000", rx=8)
    rect(ESP_X0 + 66, 85, 68, 150, "#b2bec3", stroke="#636e72", rx=4)
    text(ESP_X0 + 100, 150, "ESP32", 13, "#2d3436", "middle", "bold", parts)
    text(ESP_X0 + 100, 170, "S3", 13, "#2d3436", "middle", layer=parts)
    rect(ESP_X0 + 75, 705, 50, 30, "#95a5a6", stroke="#555", rx=3)
    text(ESP_X0 + 100, 765, "USB（書き込み時は主電源 OFF）", 12, "#555", "middle")
    for i, n in enumerate(ESP_L):
        y = ESP_Y0 + i * ESP_DY
        on = n in used_l
        label = n if not n.isdigit() else f"GPIO{n}"
        text(ESP_X0 + 12, y + 4, label, 11, "#fff" if on else "#777", weight="bold" if on else "normal",
             layer=parts)
        pin(ESP_X0, y) if on else parts.append(f'<circle cx="{ESP_X0}" cy="{y}" r="3.5" fill="#555"/>')
    for i, n in enumerate(ESP_R):
        y = ESP_Y0 + i * ESP_DY
        on = n in used_r
        label = n if not n.isdigit() else f"GPIO{n}"
        text(ESP_X1 - 12, y + 4, label, 11, "#fff" if on else "#777", "end",
             weight="bold" if on else "normal", layer=parts)
        pin(ESP_X1, y) if on else parts.append(f'<circle cx="{ESP_X1}" cy="{y}" r="3.5" fill="#555"/>')


def drv():
    x, y, w, h = 1100, 360, 200, 280
    rect(x, y, w, h, "#2c7a3f", stroke="#145a25")
    rect(x + 75, y + 110, 50, 50, "#222", stroke="#000", rx=3)
    text(x + 100, y + 80, "AE-DRV8835-S", 15, "#fff", "middle", "bold", parts)
    text(x + 100, y + 190, "PHASE/ENABLE", 11, "#d5f5e3", "middle", layer=parts)
    for name, px, py, anchor, dx, dy in (
        ("VCC", 1240, y, "middle", 0, 18), ("MODE", 1270, y, "middle", 0, 30),
        ("VM", x, 400, "start", 10, 4), ("GND", x, 470, "start", 10, 4),
        ("AIN1", x + w, 500, "end", -10, 4), ("AIN2", x + w, 530, "end", -10, 4),
        ("BIN1", x + w, 560, "end", -10, 4), ("BIN2", x + w, 590, "end", -10, 4),
        ("AOUT1", 1130, y + h, "middle", 0, -10), ("AOUT2", 1165, y + h, "middle", 0, -24),
        ("BOUT1", 1230, y + h, "middle", 0, -10), ("BOUT2", 1265, y + h, "middle", 0, -24),
    ):
        text(px + dx, py + dy, name, 10, "#fff", anchor, layer=parts)
        pin(px, py)


MOTOR_PINS = ["GND", "M1", "M2", "VCC", "C1", "C2"]


def motor(x0, label):
    """x0 = first connector pin x. Pins sit on the top edge at y=800."""
    bx = x0 - 15
    rect(bx, 800, 190, 26, "#1e8449", stroke="#0b5345", rx=3)
    for i, n in enumerate(MOTOR_PINS):
        text(x0 + i * 30, 819, n, 9, "#fff", "middle", layer=parts)
        pin(x0 + i * 30, 800)
    rect(bx + 20, 826, 150, 46, "url(#can)", stroke="#555", rx=8)
    rect(bx + 50, 872, 90, 34, "#d4ac0d", stroke="#7d6608", rx=3)
    parts.append(f'<rect x="{bx + 88}" y="906" width="14" height="22" fill="#bbb" stroke="#555"/>')
    text(bx + 95, 950, label, 14, anchor="middle", weight="bold")
    text(bx + 95, 968, "N20 6 V 30:1 encoder 付き", 11, "#555", "middle")


def servo():
    x, y = 1525, 850
    rect(x, y, 130, 100, "#2471a3", stroke="#154360", rx=6)
    parts.append(f'<circle cx="{x + 95}" cy="{y + 50}" r="22" fill="#fdfefe" stroke="#999"/>')
    parts.append(f'<rect x="{x + 88}" y="{y - 10}" width="14" height="70" rx="7" fill="#fdfefe" stroke="#999"/>')
    text(x + 65, y + 125, "サーボ Smraza S51", 14, anchor="middle", weight="bold")
    text(x + 65, y + 142, "起動時 1 A 超", 11, "#555", "middle")
    for name, py, c in (("黄 信号", 880, "#f1c40f"), ("赤 +", 905, "#e74c3c"), ("茶 −", 930, "#6e2c00")):
        parts.append(f'<line x1="1500" y1="{py}" x2="{x}" y2="{py}" stroke="{c}" stroke-width="5"/>')
        text(x + 6, py + 4, name, 10, "#fff", layer=parts)
        pin(1500, py)


def resistor(x, y1, label):
    """Vertical resistor from y1 to y1+60 at x."""
    parts.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y1 + 60}" stroke="#999" stroke-width="2.5"/>')
    rect(x - 9, y1 + 12, 18, 36, "#e6c79c", stroke="#8d6e4b", rx=7)
    for i, c in enumerate(("#8b4513", "#000", "#ff8c00")):
        parts.append(f'<rect x="{x - 9}" y="{y1 + 18 + i * 8}" width="18" height="4" fill="{c}"/>')
    text(x + 16, y1 + 34, label, 11)


def cap(x, ytop, ybot, label, kind="elec"):
    """Planned capacitor, dashed outline. + leg at ytop, − leg at ybot."""
    mid = (ytop + ybot) / 2
    parts.append(f'<line x1="{x}" y1="{ytop}" x2="{x}" y2="{ybot}" stroke="#999" stroke-width="2.5" '
                 f'stroke-dasharray="4 3"/>')
    if kind == "elec":
        parts.append(f'<rect x="{x - 11}" y="{mid - 15}" width="22" height="30" rx="4" fill="#1a1a2e" '
                     f'stroke="#ff9f1a" stroke-width="2.5" stroke-dasharray="5 3"/>')
        parts.append(f'<rect x="{x + 4}" y="{mid - 15}" width="5" height="30" fill="#bbb"/>')
        text(x - 15, mid - 8, "+", 12, "#e74c3c", "end", "bold")
    else:
        parts.append(f'<ellipse cx="{x}" cy="{mid}" rx="10" ry="8" fill="#e1a95f" stroke="#ff9f1a" '
                     f'stroke-width="2.5" stroke-dasharray="4 3"/>')
    text(x, ybot + 16 if kind == "elec" else ybot + 14, label, 11, "#d35400", "middle", "bold")


# ---------------------------------------------------------------- layout
GND_Y = 1080

battery()
fuse(250, 520)
toggle(390, 520, "スイッチ 1")
toggle(510, 520, "スイッチ 2")
buck(700, 90, "#1 制御用", "5 V", COL["v5"])
buck(700, 440, "#2 モータ用", "6 V", COL["v6"])
buck(700, 790, "#3 サーボ用", "5 V", COL["v5"])
esp32({"3V3", "4", "5", "6", "7", "15", "16", "17", "18", "13", "5V"}, {"1", "GND"})
drv()
motor(950, "左モータ")
motor(1230, "右モータ")
servo()
for i, y in enumerate((60, 120, 180)):
    resistor(1790, y, "10 kΩ")
resistor(1790, 320, "10 kΩ")
text(1802, 50, "電池電圧の分圧", 12, weight="bold")

# GND bus (the battery − terminal strip)
wire([(190, 620), (215, 620), (215, GND_Y), (1900, GND_Y)], COL["gnd"])
text(1900, GND_Y + 26, "− 列（電池 − に集約）", 13, anchor="end", weight="bold")

# battery + -> fuse -> SW1 -> SW2 -> + bus
wire([(190, 520), (250, 520)], COL["pos"])
wire([(350, 520), (390, 520)], COL["pos"])
wire([(470, 520), (510, 520)], COL["pos"])
wire([(590, 520), (640, 520)], COL["pos"])
wire([(640, 130), (640, 830)], COL["pos"])
for y in (130, 480, 830):
    wire([(640, y), (700, y)], COL["pos"])
    dot(640, y, COL["pos"])
dot(640, 520, COL["pos"])
# switch LED grounds
for x in (430, 550):
    wire([(x, 560), (x, GND_Y)], COL["gnd"])
    dot(x, GND_Y, COL["gnd"])
dot(215, GND_Y, COL["gnd"])

# buck IN−
for y, x in ((170, 664), (520, 676), (870, 688)):
    wire([(700, y), (x, y), (x, GND_Y)], COL["gnd"])
    dot(x, GND_Y, COL["gnd"])
# buck OUT−
for y, x in ((170, 925), (520, 935), (870, 945)):
    wire([(900, y), (x, y), (x, GND_Y)], COL["gnd"])
    dot(x, GND_Y, COL["gnd"])

# #1 OUT+ -> ESP32 5V
p5 = esp_l("5V")
wire([(900, 130), (1000, 130), (1000, 45), (1475, 45), (1475, p5[1]), p5], COL["v5"])
text(1240, 68, "制御 5 V → ESP32 5V", 12, COL["v5"], "middle", "bold")
cap(965, 130, 190, "470 µF")
wire([(965, 190), (925, 190)], COL["gnd"], dashed=True)
dot(965, 130, COL["v5"])
dot(925, 190, COL["gnd"])

# #2 OUT+ -> DRV VM ; DRV GND -> #2 OUT−
wire([(900, 480), (950, 480), (950, 400), (1100, 400)], COL["v6"])
wire([(1100, 470), (990, 470), (990, 560), (935, 560)], COL["gnd"])
dot(935, 560, COL["gnd"])
cap(1022, 400, 470, "470 µF")
cap(1076, 400, 470, "0.1 µF", "cer")
for x in (1022, 1076):
    dot(x, 400, COL["v6"])
    dot(x, 470, COL["gnd"])

# ESP32 3V3 -> DRV VCC + MODE
p33 = esp_l("3V3", 0)
wire([(1240, 360), (1240, p33[1]), p33], COL["v33"])
wire([(1270, 360), (1270, 335), (1240, 335)], COL["v33"])
dot(1240, 335, COL["v33"])
text(1250, p33[1] - 8, "3V3", 12, COL["v33"], weight="bold")

# GPIO4/5/6/7 -> AIN1/AIN2/BIN1/BIN2
for (g, dy, x, c) in (("4", 500, 1340, "pwm1"), ("5", 530, 1352, "pwm2"),
                      ("6", 560, 1364, "pwm3"), ("7", 590, 1376, "pwm4")):
    p = esp_l(g)
    wire([(1300, dy), (x, dy), (x, p[1]), p], COL[c])

# DRV outputs -> motors
wire([(1130, 640), (1130, 690), (980, 690), (980, 800)], COL["mot"])
wire([(1165, 640), (1165, 705), (1010, 705), (1010, 800)], "#34495e")
wire([(1230, 640), (1230, 690), (1260, 690), (1260, 800)], COL["mot"])
wire([(1265, 640), (1265, 675), (1290, 675), (1290, 800)], "#34495e")

# encoder GND -> − bus
wire([(950, 800), (950, 785), (925, 785)], COL["gnd"])
dot(925, 785, COL["gnd"])
wire([(1230, 800), (1230, 785), (1205, 785), (1205, GND_Y)], COL["gnd"])
dot(1205, GND_Y, COL["gnd"])

# encoder VCC -> ESP32 3V3 (second pin)
p33b = esp_l("3V3", 1)
wire([(1040, 800), (1040, 725), (1392, 725), (1392, p33b[1]), p33b], COL["v33"])
wire([(1320, 800), (1320, 725)], COL["v33"])
dot(1320, 725, COL["v33"])

# encoder C1/C2 -> GPIO15..18
for (px, hy, vx, g, c) in ((1070, 740, 1404, "15", "enc1"), (1100, 755, 1416, "16", "enc2"),
                           (1350, 770, 1428, "17", "enc3"), (1380, 785, 1440, "18", "enc4")):
    p = esp_l(g)
    wire([(px, 800), (px, hy), (vx, hy), (vx, p[1]), p], COL[c])

# servo
p13 = esp_l("13")
wire([(1500, 880), (1460, 880), (1460, p13[1]), p13], COL["servo"])
wire([(900, 830), (915, 830), (915, 1000), (1470, 1000), (1470, 905), (1500, 905)], COL["v5"])
wire([(1500, 930), (1482, 930), (1482, GND_Y)], COL["gnd"])
dot(1482, GND_Y, COL["gnd"])
cap(1420, 1000, GND_Y, "470 µF")
dot(1420, 1000, COL["v5"])
dot(1420, GND_Y, COL["gnd"])
text(1090, 1022, "サーボ 5 V", 12, COL["v5"], "middle", "bold")

# ESP32 GND (right header)
pg = esp_r("GND", 2)
wire([pg, (1725, pg[1]), (1725, GND_Y)], COL["gnd"])
dot(1725, GND_Y, COL["gnd"])

# battery divider: + bus -> 3 x 10k -> mid -> 10k -> GND ; mid -> GPIO1
wire([(640, 130), (640, 25), (1790, 25), (1790, 60)], COL["pos"])
dot(640, 130, COL["pos"])
wire([(1790, 240), (1790, 320)], COL["adc"])
wire([(1790, 380), (1790, GND_Y)], COL["gnd"])
dot(1790, GND_Y, COL["gnd"])
p1 = esp_r("1")
wire([(1790, 285), (1745, 285), (1745, p1[1]), p1], COL["adc"])
dot(1790, 285, COL["adc"])
text(1800, 274, "中点 8.4 V → 2.1 V", 11, COL["adc"], weight="bold")
cap(1880, 285, 360, "0.1 µF", "cer")
wire([(1790, 285), (1880, 285)], COL["adc"], dashed=True)
wire([(1880, 360), (1880, GND_Y)], COL["gnd"], dashed=True)
dot(1880, GND_Y, COL["gnd"])

# title + legend
text(40, 60, "猫じゃらしロボ 本番配線", 28, weight="bold")
text(40, 90, "2026-10-15 時点。ユーザー報告の配線に、旧設計のコンデンサと分圧回路を重ねた図", 13, "#555")
text(40, 112, "実機での通電確認はしていない", 13, "#555")
lg = [
    ("実線", "報告済みの配線"),
    ("点線＋橙枠", "コンデンサ（未取り付け。電解は帯の側を − に）"),
    ("分圧抵抗", "取り付け済みかは未確認"),
    ("ピン位置", "ESP32 は DevKitC-1 の並び、DRV8835・encoder は印字に従う"),
]
for i, (k, v) in enumerate(lg):
    text(40, 160 + i * 24, f"{k}：", 13, weight="bold")
    text(140, 160 + i * 24, v, 13)
text(40, GND_Y + 70, "スイッチ 1 と 2 は直列。どちらを OFF にしても全体が止まる", 13, "#555")

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
<pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="1" fill="#d0d4d8"/></pattern>
<linearGradient id="lipo" x1="0" x2="1"><stop offset="0" stop-color="#cfd8dc"/>
<stop offset="0.5" stop-color="#f5f7f8"/><stop offset="1" stop-color="#b0bec5"/></linearGradient>
<linearGradient id="can" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ecf0f1"/>
<stop offset="0.5" stop-color="#bdc3c7"/><stop offset="1" stop-color="#95a5a6"/></linearGradient>
</defs>
<rect width="100%" height="100%" fill="#f7f7f4"/>
<rect width="100%" height="100%" fill="url(#grid)"/>
{chr(10).join(parts)}
{chr(10).join(wires)}
{chr(10).join(tops)}
</svg>
'''
(OUT / "wiring.svg").write_text(svg, encoding="utf-8")
html = OUT / "_render.html"
html.write_text(f'<html><body style="margin:0">{svg}</body></html>', encoding="utf-8")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                f"--window-size={W},{H}", "--force-device-scale-factor=1.5",
                f"--screenshot={OUT / 'wiring.png'}", html.as_uri()], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
html.unlink()
print("wrote", OUT / "wiring.svg", OUT / "wiring.png")
