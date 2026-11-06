"""Erzeugt Demo-GIFs aus den Szenenbildern + solution.json (nur Visualisierung).

Aufruf (im Projektordner, in dem main.py liegt):
    pip install pillow
    python3 render_gifs.py . docs

Ablauf:
  1) main.py wird als Modul geladen -> die vorhandenen Detektoren
     (detect_stop_light, kreuzung_blockiert, myCar_steht, ...) werden
     wiederverwendet, es gibt KEINE neue Erkennungslogik.
  2) Pro Szene werden alle Bilder geladen und die Detektoren erneut
     ausgeführt, um Suchbereiche, erkannte Fahrzeuge (Bounding-Box) und
     das Differenzbild einzuzeichnen.
  3) Die Entscheidung (Geschwindigkeit + Richtung) kommt aus solution.json
     und wird oben als Kopfleiste eingeblendet.
  4) Alle Bilder einer Szene werden mit Pillow zu einem GIF zusammengefügt
     (900 ms pro Bild), zusätzlich ein GIF mit allen Szenen.
"""
import sys, os, json, importlib.util
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if len(sys.argv) >= 3:
    proj, out = sys.argv[1], sys.argv[2]
else:
    # Ohne Argumente (z. B. per "Run"-Knopf in VS Code): vom Skript aus nach
    # oben den Ordner mit main.py suchen und die GIFs nach <projekt>/docs schreiben
    proj = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(proj, "main.py")):
        parent = os.path.dirname(proj)
        if parent == proj:
            sys.exit("main.py nicht gefunden. Aufruf: python3 render_gifs.py <projektordner> <ausgabeordner>")
        proj = parent
    out = os.path.join(proj, "docs")
print("Projekt:", proj, "-> GIFs nach:", out)
os.makedirs(out, exist_ok=True)

spec = importlib.util.spec_from_file_location("trabi_main", os.path.join(proj, "main.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Bilderordner suchen: "images" direkt im Projekt oder in einem Unterordner
# (z. B. "DBV Projekt/images"), egal wie der Unterordner heißt
IMG_DIR = None
for root, dirs, _ in os.walk(proj):
    if os.path.basename(root) == "images" and "Scene_1" in dirs:
        IMG_DIR = root
        break
if IMG_DIR is None:
    sys.exit("Ordner 'images' mit Scene_1 ... Scene_5 nicht gefunden.")
print("Bilder aus:", IMG_DIR)
sol = json.load(open(os.path.join(proj, "solution.json"), encoding="utf-8"))

TITLES = {
    1: "Szene 1 · Ampel, dann rechts abbiegen",
    2: "Szene 2 · Vorfahrtstraße, durchfahren",
    3: "Szene 3 · LKW quert, dann links abbiegen",
    4: "Szene 4 · Auto quert vor dem Trabi",
    5: "Szene 5 · Rechts vor Links",
}
W_OUT = 960
# Schriftarten: je nach Betriebssystem liegen sie woanders -> erste gefundene nehmen
BOLD_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",      # macOS
    "/Library/Fonts/Arial Bold.ttf",                          # macOS (älter)
    "C:/Windows/Fonts/arialbd.ttf",                           # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Linux
]
REGULAR_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(candidates, size):
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)   # Notlösung: Pillow-eigene Schrift


font_t = load_font(BOLD_FONTS, 24)
font_s = load_font(REGULAR_FONTS, 18)
font_big = load_font(BOLD_FONTS, 34)
font_dir = load_font(BOLD_FONTS, 22)
ARROW = {"geradeaus": "↑ geradeaus", "rechts": "→ rechts", "links": "← links"}


def frac_rect(img, fx0, fy0, fx1, fy1, color, label=None):
    h, w = img.shape[:2]
    p0, p1 = (int(fx0 * w), int(fy0 * h)), (int(fx1 * w), int(fy1 * h))
    cv2.rectangle(img, p0, p1, color, 4)
    if label:
        cv2.putText(img, label, (p0[0] + 8, p0[1] + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4)


def diff_blobs(img, plate, min_area):
    d = cv2.cvtColor(cv2.absdiff(img, plate), cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(d, 35, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, _, st, cent = cv2.connectedComponentsWithStats(th, 8)
    return th, [(st[k], cent[k]) for k in range(1, n) if st[k][4] >= min_area]


def steht_list(frames):
    grays = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for _, i in frames]
    h, w = grays[0].shape
    n = len(frames)
    res = []
    for i in range(n):
        p = i > 0 and m.myCar_steht(grays[i - 1], grays[i], h, w)
        q = i < n - 1 and m.myCar_steht(grays[i], grays[i + 1], h, w)
        res.append(bool(p or q))
    return res


def annotate(scene, frames):
    """Gibt pro Frame (bild_mit_overlay, optionales_differenzbild, info) zurück."""
    out_frames = []
    plate, steht = None, None
    if scene in (4, 5):
        steht = steht_list(frames)
        idx = [i for i in range(len(frames)) if steht[i]]
        if scene == 4:
            src = idx or list(range(len(frames)))
            plate = np.median(np.stack([frames[i][1] for i in src]).astype(np.float32), axis=0).astype(np.uint8)
        else:
            plate = frames[idx[-1]][1] if idx else frames[-1][1]

    for i, (name, img) in enumerate(frames):
        vis = img.copy()
        diff, info = None, ""
        h, w = img.shape[:2]
        if scene == 1:
            state, near = m.detect_stop_light(img)
            info = f"Ampel: {state or 'grün/frei'}" + (" (nah)" if near else "")
            frac_rect(vis, 0.55, 0.0, 1.0, 0.75, (0, 200, 255), "Suchbereich Ampel")
        elif scene == 2:
            info = "Vorfahrtsschild erkannt" if m.detect_vorfahrtstrasse(img) else "freie Fahrt"
        elif scene == 3:
            blk = m.kreuzung_blockiert(img)
            info = "Hindernis voraus!" if blk else "Weg frei"
            frac_rect(vis, 0.30, 0.42, 0.70, 0.60, (0, 0, 255) if blk else (0, 200, 0), "Zone vor dem Auto")
        elif scene == 4:
            diff, blobs = diff_blobs(img, plate, 8000)
            frac_rect(vis, 0.40, 0.44, 0.80, 0.75, (0, 200, 0), "Zone vor dem Auto")
            if steht[i]:
                for (x, y, bw, bh, a), _ in blobs:
                    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 5)
            info = ("Trabi steht" if steht[i] else "Trabi fährt")
        elif scene == 5:
            diff, blobs = diff_blobs(img, plate, 3000)
            frac_rect(vis, 0.55, 0.42, 1.0, 0.62, (0, 200, 0), "Verkehr von rechts")
            if steht[i]:
                for (x, y, bw, bh, a), (cx, cy) in blobs:
                    if x + bw > 0.55 * w and 0.42 * h < cy < 0.62 * h:
                        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 5)
            info = ("Trabi steht" if steht[i] else "Trabi fährt")
        out_frames.append((name, vis, diff, info))
    return out_frames


def compose(scene, name, vis, diff, info, entry):
    scale = W_OUT / vis.shape[1]
    small = cv2.resize(vis, (W_OUT, int(vis.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if diff is not None:
        dw = W_OUT // 4
        dsm = cv2.resize(diff, (dw, int(diff.shape[0] * dw / diff.shape[1])), interpolation=cv2.INTER_AREA)
        dsm = cv2.cvtColor(dsm, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dsm, (0, 0), (dsm.shape[1] - 1, dsm.shape[0] - 1), (255, 255, 255), 2)
        y0 = small.shape[0] - dsm.shape[0] - 12
        small[y0:y0 + dsm.shape[0], W_OUT - dw - 12:W_OUT - 12] = dsm
    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    bar_h = 84
    canvas = Image.new("RGB", (W_OUT, pil.height + bar_h), (24, 24, 32))
    canvas.paste(pil, (0, bar_h))
    d = ImageDraw.Draw(canvas)
    d.text((16, 14), TITLES[scene], font=font_t, fill=(255, 255, 255))
    d.text((16, 48), f"Bild {name}  ·  {info}", font=font_s, fill=(190, 190, 205))
    # Entscheidung rechts in der Kopfleiste: Richtung + Geschwindigkeits-Schild
    v = entry["geschwindigkeit"]
    r = 34
    cx, cy = W_OUT - 16 - r, bar_h // 2
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255), outline=(220, 30, 30), width=7)
    tw = d.textlength(str(v), font=font_big)
    d.text((cx - tw / 2, cy - 24), str(v), font=font_big, fill=(20, 20, 20))
    lab = ARROW.get(entry["richtung"], entry["richtung"])
    lw = d.textlength(lab, font=font_dir)
    d.text((cx - r - 14 - lw, cy - 14), lab, font=font_dir, fill=(255, 255, 255))
    if diff is not None:
        dw = W_OUT // 4
        dh = int(diff.shape[0] * dw / diff.shape[1])
        d.text((W_OUT - dw - 12, canvas.height - dh - 12 - 22), "Differenzbild", font=font_s, fill=(255, 255, 255))
    return canvas


all_frames = []
for s in range(1, 6):
    frames = m.load_scene(os.path.join(IMG_DIR, f"Scene_{s}"))
    entries = {e["bildname"]: e for e in sol[f"Scene_{s}"]}
    imgs = [compose(s, n, v, df, inf, entries[n]) for n, v, df, inf in annotate(s, frames)]
    durs = [900] * len(imgs)
    durs[-1] = 1600
    imgs[0].save(os.path.join(out, f"scene_{s}.gif"), save_all=True, append_images=imgs[1:],
                 duration=durs, loop=0, optimize=True)
    all_frames += list(zip(imgs, durs))
    print("scene", s, len(imgs))

ims, durs = zip(*all_frames)
ims[0].save(os.path.join(out, "trabi_alle_szenen.gif"), save_all=True, append_images=list(ims[1:]),
            duration=list(durs), loop=0, optimize=True)
print("done")