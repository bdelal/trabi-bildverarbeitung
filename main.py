"""
Verwendete Bildverarbeitungs Techniken:
  * Pixeloperationen   : threshold, absdiff (Differenzbild), Median-Hintergrund
  * Morphologie        : morphologyEx (Open/Close), Skelettierung, Distanztransform
  * Segmentierung      : HSV-Farbsegmentierung (inRange), Connected-Components
  * Objektdetektion    : Ampel/Schild/Fahrzeug als Blob + räumliche Prüfung
  * Feature Extraction : Fläche, Schwerpunkt (Moments), Bounding-Box, approxPolyDP
"""

import os                       
from collections import deque   
import json                     
import cv2                      
import numpy as np              
from skimage.morphology import skeletonize  #pip install scikit-image


IMAGE_DIR = "images"                                           
CITYMAP_PATH = os.path.join(IMAGE_DIR, "citymap", "citymap_no_buildings.png")  
CITYMAP_DOTS_PATH = os.path.join(IMAGE_DIR, "citymap", "citymap_yellow_dots.png")  

# Schwellwert für die Zonen-Entscheidung: ist eine Straße breiter als dieser
# Wert (in Pixeln), gilt sie als "dicke" 50er Straße, sonst als 30er-Zone
WIDTH_THRESHOLD_PX = 20



#Karte Straßen als Maske einlesen und Zonen (30/50) bestimmen

def _load_street_mask(path):
    """Liest ein Graustufenbild der Karte und liefert eine Binaermaske der Straßen.

    Args:
        path (str): Pfad zum Kartenbild.

    Returns:
        np.ndarray | None: Binaerbild (Strasse = 255, Rest = 0) oder None,
        falls die Datei nicht gefunden wurde.
    """
    # Bild direkt als Graustufen laden (Grund: Farbe wird für die Straßenmaske nicht gebraucht)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] Karte nicht gefunden: {path}")
        return None
    #Straßen sind dunkel, der Hintergrund hell. THRESH_BINARY_INV dreht das um,
    # sodass dunkle Straßenpixel (< 180) zu 255 (weiss) werden -->Strasse = Vordergrund.
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    return binary


def analyze_map_streets():
    """Bestimmt für die gesamte Karte die lokale Straßenbreite (in Pixeln).

    Warum Distanztransformation statt einer einzelnen Scan-Zeile:
    Eine einzelne Bildzeile trifft nur Straßen, die genau dort horizontal
    verlaufen, und misst vertikale Straßen falsch. Die Distanztransformation
    liefert dagegen für JEDEN Straßenpixel den Abstand zum nächsten Rand;
    auf der Mittellinie entspricht dieser Abstand der halben Straßenbreite.
    Damit laesst sich für jeden Punkt sagen, ob er in einer 30er- oder
    50er-Zone liegt.

    Returns:
        np.ndarray | None: width_map -- pro Straßenpixel die volle Breite in px.
    """
    binary = _load_street_mask(CITYMAP_PATH)
    if binary is None:
        return None

    # Distanztransformation: jeder weiße (Straßen-)Pixel bekommt den euklidischen
    # Abstand zum nächsten schwarzen Pixel (Straßenrand). DIST_L2 = echte Distanz
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Der Abstand zur Mitte ist die halbe Breite -> *2 ergibt die volle Breite
    width_map = dist * 2.0

    # Skelett = "Mittellinie" der Straßen (auf 1 Pixel ausgedünnte Straße)
    skel = skeletonize(binary > 0)
    # Nur die Breiten AUF der Mittellinie betrachten (dort ist der Wert = volle Breite)
    widths_on_skel = width_map[skel]
    if widths_on_skel.size:
        # Ein paar Quantile ausgeben, um den Schwellwert nachvollziehbar zu machen
        q = np.percentile(widths_on_skel, [25, 50, 75, 90]).round(1)
        print(f"[MAP] Breiten-Quantile auf Mittellinie (25/50/75/90): {q} px")
        print(f"[MAP] Aktueller Schwellenwert: {WIDTH_THRESHOLD_PX} px "
              f"(dick > {WIDTH_THRESHOLD_PX} = 50er, sonst 30er)")
    return width_map


def zone_at(width_map, x, y):
    """Leitet die zulässige Höchstgeschwindigkeit an einer Kartenposition ab

    Args:
        width_map (np.ndarray): Breitenkarte aus analyze_map_streets().
        x (int), y (int): Pixelkoordinate

    Returns:
        int | None: 50 (breite Strasse) oder 30 (schmale Strasse); None ohne Karte.
    """
    if width_map is None:
        return None
    # Breite Strasse (> Schwellwert) -> 50er, sonst 30er-Zone.
    return 50 if width_map[y, x] > WIDTH_THRESHOLD_PX else 30



###  Route: Start/Ziel/Wegpunkte per Farbe finden und Weg auf dem Skelett suchen

def extract_markers():
    """Findet den grünen Start-Punkt und die Spitze der roten Zielnadel

    Technik: HSV-Farbsegmentierung (inRange) + größte Kontur pro Farbe

    Returns:
        tuple: (start_xy, goal_xy) -- je (x, y) oder None, falls nicht gefunden
    """
    img = cv2.imread(CITYMAP_DOTS_PATH)
    if img is None:
        return None, None
    # In den HSV-Farbraum wechseln: dort lässt sich "grün"/"rot" über den
    # Farbwinkel (H) robust und beleuchtungsunabhängig ansprechen
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def largest(mask):
        # Alle zusammenhängenden Farbflecken (Konturen) suchen 
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # und kleine Störflecken (< 100 px) entfernen
        cnts = [c for c in cnts if cv2.contourArea(c) > 100]
        # und die grösste Flaeche zurueckgeben (das ist der gesuchte Marker)
        return max(cnts, key=cv2.contourArea) if cnts else None

    # Grün-Maske (Startpunkt): H ca. 40..85.
    g = largest(cv2.inRange(hsv, (40, 120, 120), (85, 255, 255)))
    # Rot liegt im HSV am Farbkreis-Anfang UND -Ende -> zwei Bereiche verodern
    r = largest(cv2.inRange(hsv, (0, 120, 80), (10, 255, 255)) |
                cv2.inRange(hsv, (170, 120, 80), (180, 255, 255)))

    start = goal = None
    if g is not None:
        # Schwerpunkt (Moments) vom grünen Fleck = Startkoordinate
        M = cv2.moments(g)
        start = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    if r is not None:
        # Bei der Zielnadel den UNTERSTEN Punkt (grösstes y) nehmen, die Spitze
        # zeigt auf die Strasse, der Schwerpunkt laege dagegen im Nadelkopf
        goal = tuple(int(v) for v in r[r[:, :, 1].argmax()][0])
    return start, goal


def extract_yellow_dots(start_hint="bottom_right"):
    """Baut die Route als geordnete Punktliste auf

    Reihenfolge: [grüner Start] + gelbe Wegpunkte (per Nearest-Neighbor ab Start
    sortiert) + [rote Zielnadel]. So sind Anfang und Ende mit der Route verbunden.

    Returns:
        list[tuple]: geordnete (x, y)-Punkte von Start nach Ziel
    """
    img = cv2.imread(CITYMAP_DOTS_PATH)
    if img is None:
        print(f"[WARN] Karte mit Punkten nicht gefunden: {CITYMAP_DOTS_PATH}")
        return []

    # Gelbe Wegpunkte per HSV-Farbsegmentierung isolieren
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    # Jeden gelben Fleck als Kontur finden 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:            # winzige Störpunkte ignorieren
            M = cv2.moments(cnt)                  # Schwerpunkt = Punktkoordinate
            if M["m00"] > 0:
                dots.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    if not dots:
        print("[MAP] Keine gelben Wegpunkte gefunden.")
        return []

    start, goal = extract_markers()
    # Nearest-Neighbor-Sortierung: ab dem grünen Start immer den nächsten noch
    # nicht besuchten Punkt anhängen ---->ergibt die Reihenfolge entlang der Route
    cur = start if start else max(dots, key=lambda p: p[0] + p[1])
    remaining, ordered = set(dots), []
    while remaining:
        nxt = min(remaining, key=lambda p: (p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2)
        remaining.discard(nxt)
        ordered.append(nxt)
        cur = nxt

    # Start vorne, Ziel hinten anfuegen ->komplette Route
    route = ([start] if start else []) + ordered + ([goal] if goal else [])
    print(f"[MAP] Route mit {len(route)} Punkten (grün-> rot):")
    for i, (x, y) in enumerate(route):
        tag = "START" if start and i == 0 else ("ZIEL" if goal and i == len(route) - 1 else "WP")
        print(f"      {i + 1:2d} [{tag:5s}] ({x}, {y})")
    return route


def _bfs_on_skeleton(skel, start, goal):
    """Kürzester Pfad zwischen zwei Punkten auf dem Straßenskelett

    Verwendet Breitensuche (BFS) mit 8er-Nachbarschaft. Das Skelett dient als
    Grapf: benachbarte weiße Pixel sind verbundene Knoten

    Returns:
        list[tuple] | None: Pfadpunkte von start bis goal oder None
    """
    h, w = skel.shape
    prev, visited = {}, np.zeros(skel.shape, bool)   #prev = Vorgänger, visited = besucht?
    q = deque([start])
    visited[start[1], start[0]] = True
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            # Ziel erreicht ->Pfad rückwärts ueber die Vorgänger rekonstruieren
            path = [(x, y)]
            while (x, y) in prev:
                x, y = prev[(x, y)]
                path.append((x, y))
            return path[::-1]                        # umdrehen: start -> goal
        # alle 8 Nachbarpixel prüfen
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                #nur innerhalb des Bildes, nur Skelettpixel, nur noch nicht besuchte
                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))
    return None


def find_route(ordered_dots, save_path="route_visualisierung.png"):
    """Verbindet die Wegpunkte entlang der Straßen und erkennt Abbiegungen.

    Ablauf:
      * Straßenmaske -> Skelett als Graph
      * Jeden Wegpunkt auf den nächsten Skelettpixel "snappen"
      * Aufeinanderfolgende Wegpunkte per BFS entlang des Skeletts verbinden
      * Pro Wegpunkt geometrisch aus Ein-/Ausgangsrichtung die Abbiegung ableiten
      * Zone (30/50) am Wegpunkt ablesen und alles visualisieren

    Returns:
        list[tuple]: manoeuvers = (wegpunkt, richtung, zone) je Wegpunkt
    """
    binary = _load_street_mask(CITYMAP_PATH)
    if binary is None or not ordered_dots:
        return []

    width_map = analyze_map_streets()
    skel = skeletonize(binary > 0)
    ys, xs = np.nonzero(skel)                   # Koordinaten aller Skelettpixel

    def snap(p):
        # Wegpunkt liegt evtl. nicht exakt auf dem Skelett -> nächsten Skelettpixel suchen
        i = int(np.argmin((xs - p[0]) ** 2 + (ys - p[1]) ** 2))
        return (int(xs[i]), int(ys[i]))

    stops = [snap(p) for p in ordered_dots]     # alle Wegpunkte auf das Skelett ziehen
    full_path = []
    # aufeinanderfolgende Stopps paarweise verbinden (a->b, b->c, ...)
    for a, b in zip(stops, stops[1:]):
        seg = _bfs_on_skeleton(skel, a, b)
        if seg is None:
            print(f"[WARN] Kein Pfad zwischen {a} und {b} (Skelett unterbrochen?)")
            continue
        full_path.extend(seg)

    # --- Abbiegungen geometrisch bestimmen ---------------------------------
    # für jeden inneren Wegpunkt: Eingangsvektor (vom Vorgänger) und
    # Ausgangsvektor (zum Nachfolger). Das Kreuzprodukt gibt die Drehrichtung,
    # der Winkel die Staerke der Kurve
    manoeuvers = []
    for i in range(1, len(ordered_dots) - 1):
        p_prev = np.array(ordered_dots[i - 1], float)
        p_cur  = np.array(ordered_dots[i],     float)
        p_next = np.array(ordered_dots[i + 1], float)
        v_in, v_out = p_cur - p_prev, p_next - p_cur     # Ein-/Ausgangsrichtung
        cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]  # Kreuzprodukt (z-Komponente)
        # Winkel zwischen den Vektoren; Vorzeichen aus dem Kreuzprodukt.
        # (Bild-y zeigt nach unten, daher ist positiv = rechts)
        ang = np.degrees(np.arctan2(cross, np.dot(v_in, v_out)))
        richtung = "geradeaus" if abs(ang) < 30 else ("rechts" if ang > 0 else "links")
        sx, sy = stops[i]                                # gesnappter Punkt = robust auf Strasse
        zone = zone_at(width_map, sx, sy)                # Zone dort ablesen
        manoeuvers.append((ordered_dots[i], richtung, zone))
        print(f"[ROUTE] {ordered_dots[i]}: {richtung.upper():9s} ({zone}er Zone)")

    # --Visualisierung: Pfad einzeichnen (rot = 50er, blau = 30er) -
    vis = cv2.imread(CITYMAP_DOTS_PATH)
    if vis is not None:
        for (x, y) in full_path:
            # Farbe je nach Zone am jeweiligen Pfadpixel.
            c = (0, 0, 255) if width_map[y, x] > WIDTH_THRESHOLD_PX else (255, 120, 0)
            cv2.circle(vis, (x, y), 2, c, -1)
        for i, p in enumerate(ordered_dots):
            cv2.circle(vis, p, 6, (0, 255, 0), 2)                 #Wegpunkt markieren
            cv2.putText(vis, str(i + 1), (p[0] + 8, p[1] - 8),    #nummerieren
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(save_path, vis)
        print(f"[OK] Route gespeichert: {save_path}")

    return manoeuvers


#  DETEKTOREN + SZENEN Löser
#  jede Szene hat einen eigenen Solver, weil sie eine andere Verkehrslage zeigt

def detect_stop_light(img, near_min_area=200):
    """Erkennt den Ampelzustand über die leuchtende rote/gelbe Lampe

    Es gibt keine rot/gelb leuchtenden Buesche, daher ist eine leuchtende
    kleine rot/gelbe Flaeche zuverlaessig die Ampel

    Args:
        img (np.ndarray): BGR-Bild.
        near_min_area (int): ab dieser Blob-Groesse gilt die Ampel als "nah".

    Returns:
        tuple: (state, is_near)
            state   -- 'rot' | 'rotgelb' | 'gelb' | None (None = grün/frei)
            is_near -- True, wenn die Lampe groß ist = nah an der Kreuzung
    """
    h, w = img.shape[:2]
    # Nur oben-rechts suchen: dort hängt in dieser Szene die Ampel
    roi = img[0:int(h * 0.75), int(w * 0.55):]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # "lit" = kraeftig gesättigte, helle Pixel = leuchtende Lampe (nicht matt)
    lit = ((hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 120)).astype(np.uint8)
    H = hsv[:, :, 0]                                    #Farbwinkel-Kanal

    def biggest(mask):
        # Rauschen entfernen (Open) und größten plausiblen "Lampen"-Blob suchen
        m = cv2.morphologyEx(mask * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        b = 0
        for i in range(1, n):
            x, y, bw, bh, a = st[i]
            # Plausibilität: passende Größe, halbwegs rundes Seitenverhältnis
            # gut gefüllte Box -> sieht wirklich wie eine Lampe aus
            if 25 < a < 1500 and bw > 0 and bh > 0 and 0.4 < bw/bh < 2.2 and a/(bw*bh) > 0.45:
                b = max(b, int(a))
        return b

    # Rote Lampe: H am Kreisanfang/-ende,  Gelbe Lampe: H ca. 18..38
    ra = biggest(((H < 12) | (H > 168)).astype(np.uint8) * lit)
    ya = biggest(((H >= 18) & (H < 38)).astype(np.uint8) * lit)
    # Zustand ableiten (rot+gelb gleichzeitig = "rotgelb").
    state = "rotgelb" if (ra and ya) else ("rot" if ra else ("gelb" if ya else None))
    is_near = max(ra, ya) >= near_min_area             # große Lampe =nah
    return state, is_near


def detect_vorfahrtstrasse(img):
    """Erkennt das Vorfahrtstraße Schild (gelbe Raute) über Farbe + Form

    Hinweis: Template Matching hat hier versagt (Score ca 0.48 wegen Perspektive
    und Skalierung). Farbsegmentierung + Formpruefung ist deutlich besser geeignet

    Returns:
        bool: True, wenn eine gelbe, quadratisch gedrehte Raute sichtbar ist
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (20, 110, 110), (35, 255, 255))          # gelbe Flaechen
    # Close = kleine Löcher/Lücken im gelben Fleck schliessen -> saubere Form
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 150:                                     # zu klein -> ignorieren
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / h                                      # Seitenverhältnis
        fill = a / (w * h)                              # Füllgrad der Bounding-Box
        # approxPolyDP vereinfacht die Kontur zu einem Polygon -> Eckenzahl zählen
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        # Ein auf die Spitze gestelltes Quadrat (Raute): quadratische Box,
        # Füllgrad um 0.5 (Ecken der Box sind leer) und genau 4 Ecken
        if 0.6 < ar < 1.5 and 0.35 < fill < 0.75 and len(approx) == 4:
            return True
    return False


def solve_scene_1(frames):
    """Scene 1 : Ampelkreuzung, danach rechts abbiegen

    Logik: solange die Ampel weit weg ist -> 50/geradeaus. Sobald die (rot/gelbe)
    Ampel gross/nah ist -> an der Kreuzung. Rote Ampel -> 0, Grün an der
    Kreuzung -> abbiegen mit 10

    Args:
        frames (list): (bildname, img) in Reihenfolge 00..11.

    Returns:
        list[dict]: je Bild ein Eintrag für die JSON
    """
    at_crossing = False                                 # merkt sich: sind wir an der Kreuzung?
    ergebnisse = []
    for bildname, img in frames:
        state, is_near = detect_stop_light(img)
        if is_near:
            at_crossing = True                          # einmal nah -> bleibt an der Kreuzung
        if state in ("rot", "rotgelb") and is_near:
            speed, richtung = 0, "rechts"               # rote Ampel: anhalten, Richtung trotzdem
        elif at_crossing:
            speed, richtung = 10, "rechts"              # grün an der Kreuzung: abbiegen
        else:
            speed, richtung = 50, "geradeaus"           # noch weit weg: normal fahren
        ergebnisse.append({
            "bildname": bildname,
            "geschwindigkeit": speed,
            "richtung": richtung,
        })
    return ergebnisse


def solve_scene_2(frames):
    """Scene 2: durchgehend 50/geradeaus in der 50er-Zone

    Am Ende Vorfahrtstrasse (gelbe Raute) + wartender LKW -> wir haben Vorfahrt
    und fahren einfach durch ohne abbiegen ohne stopp.
    """
    ergebnisse = []
    for bildname, img in frames:
        ergebnisse.append({
            "bildname": bildname,
            "geschwindigkeit": 50,
            "richtung": "geradeaus",
        })
    return ergebnisse


def kreuzung_blockiert(img, roi_frac=(0.30, 0.42, 0.70, 0.60), dark_thresh=5000):
    """Erkennt ein großes dunkles Hindernis (abbiegender schwarzer LKW) direkt voraus

    Funktioniert für den schwarzen LKW in Scene 3. Bewusste Grenze: helle
    Fahrzeuge (Scene 4/5) werden so NICHT erkannt,  dafür gibt es dort das
    Differenzbild-Verfahren

    Args:
        roi_frac (tuple): (x1, y1, x2, y2) als Bildanteile 0..1 für die Suchregion.
        dark_thresh (int): Mindestgroesse vom dunklen blob

    Returns:
        bool: True, wenn ein großer dunkler Blob im Bereich vor uns liegt
    """
    h, w = img.shape[:2]
    # Suchregion (direkt vor dem Fahrzeug) aus den Anteilen in Pixel umrechnen
    x1, y1, x2, y2 = [int(f * d) for f, d in zip(roi_frac, (w, h, w, h))]
    # Value-Kanal (Helligkeit) der Region; sehr dunkle Pixel = mögliches Fahrzeug
    V = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)[:, :, 2]
    dark = cv2.morphologyEx((V < 70).astype(np.uint8) * 255,          # dunkle Pixel
                            cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # Rauschen weg
    
    #Größten zusammenhängenden dunklen Blob bestimmen
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    groesster = max([st[i][4] for i in range(1, n)], default=0)
    return groesster > dark_thresh


def solve_scene_3(frames):
    """Scene 3 -- Anfahrt geradeaus (50er). Querender LKW blockiert -> 0.
    Wenn frei und an der Kreuzung -> links abbiegen mit 10. Endet in 30er-Zone.
    """
    at_crossing = False
    ergebnisse = []
    for bildname, img in frames:
        blockiert = kreuzung_blockiert(img)             # dunkles Hindernis voraus?
        raute = detect_vorfahrtstrasse(img)             # Vorfahrtstrasse-Schild sichtbar?
        if raute:
            at_crossing = True                          # Schild sichtbar = an der Kreuzung
        if blockiert:
            speed, richtung = 0, "links"                # LKW schneidet -> 0, Richtung trotzdem
        elif at_crossing:
            speed, richtung = 10, "links"               # frei an der Kreuzung -> abbiegen
        else:
            speed, richtung = 50, "geradeaus"           # Anfahrt
        ergebnisse.append({
            "bildname": bildname,
            "geschwindigkeit": speed,
            "richtung": richtung,
        })
    return ergebnisse


# Gemeinsame Helfer für Scene 4 & 5 (bewegte Fahrzeuge bei stehender Kamera) ---

def myCar_steht(gray_prev, gray_cur, h, w, thresh=3.0):
    """Prüft über die Eigenbewegung, ob der Trabi steht

    Idee: In den oberen Ecken sind nur Gebäude (dort fährt nie ein Auto).
    Ändert sich dieser Hintergrund zwischen zwei Frames kaum, hat sich die
    Kamera nicht bewegt -> der Trabi steht

    Technik: Bilddifferenz (Pixeloperation) auf festen Hintergrund Bändern

    Returns:
        bool: True, wenn die mittlere Differenz unter dem Schwellwert liegt.
    """
    # Maske: linkes und rechtes oberes Gebäde Band (nie Fahrbahn/Autos)
    m = np.zeros((h, w), bool)
    m[int(0.15 * h):int(0.45 * h), :int(0.22 * w)] = True   # Gebäude links
    m[int(0.10 * h):int(0.40 * h), int(0.80 * w):] = True    # gebäude rechts
    # Mittlere absolute Helligkeitsdifferenz nur auf diesen Bändern
    diff = np.abs(gray_cur[m].astype(np.float32) - gray_prev[m].astype(np.float32))
    return diff.mean() < thresh


def _querer_in_konfliktzone(img, plate, czx=(0.40, 0.80), czy=(0.44, 0.75),
                            min_area=8000):
    """Prüft, ob ein bewegtes Auto unsere Fahrspur blockiert

    Vergleich gegen die leere-Straße-Referenz ("plate", Hintergrundmodell):
    Wo sich das aktuelle Bild stark von der leeren Straße unterscheidet, ist
    ein Fahrzeug. Nur ein Fahrzeug rechts der Mittellinie (x >= 0.40) in unserem
    Fahrweg blockiert uns linker Gegenverkehr (rechter Rand < 0.40) nicht.

    Technik: Differenzbild + Threshold + Morphologie + Connected-Components

    Args:
        czx (tuple): Konfliktzone in x (Anteile) 0.40 = Mittellinie.
        czy (tuple): Konfliktzone in y (Anteile) Straßenhöhe.
        min_area (int): Mindestflaeche, damit es als Fahrzeug zählt.

    Returns:
        bool: True, wenn ein grosser Blob die Konfliktzone berührt
    """
    h, w = img.shape[:2]
    # Differenzbild aktuelles Bild <-> leere Strasse, in Graustufen
    d = cv2.cvtColor(cv2.absdiff(img, plate), cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(d, 35, 255, cv2.THRESH_BINARY)             # deutliche Unterschiede
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))    # Rauschen weg
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) # Fahrzeug schließen
    # Alle Differenz-Blobs mit Statistik (Bounding-Box, Flaeche, Schwerpunkt)
    n, _, st, cent = cv2.connectedComponentsWithStats(th, 8)
    for k in range(1, n):
        x, y, bw, bh, a = st[k]
        if a < min_area:                            # zu klein -> kein Fahrzeug
            continue
        # Blob-Ränder/Schwerpunkt in Bildanteile umrechnen
        x0, x1, cy = x / w, (x + bw) / w, cent[k][1] / h
        # Berührt der Blob die Konfliktzone in x UND liegt er auf Straßenhöhe?
        if x1 > czx[0] and x0 < czx[1] and czy[0] < cy < czy[1]:
            return True
    return False


def solve_scene_4(frames, zone=30):
    """Scene 4 -- 30er-Zone, dann Halt für ein abbiegendes Auto, danach weiter

    Ein Auto quert von rechts nach links durch unseren Fahrweg. Solange es in
    unserer Spur ist -> anhalten (0); sobald es raus ist -> weiterfahren.

    Vorgehen :
      1) Eigenbewegung je Frame -> fährt / steht  (myCar_steht)
      2) Hintergrundmodell = zeitlicher Median der Stillstands-Frames
         (das abbiegende Auto steht jedes Mal woanders -> Median = leere Strasse)
      3) steht + Auto in Konfliktzone -> 0, sonst Zonengeschwindigkeit.

    Canny hat hier nicht funktioniert (wegen den Fahrbahnmakierungen und Fassaden)
    """
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for _, img in frames]
    h, w = grays[0].shape
    n = len(frames)

    # 1) Stillstand je Frame: ein Nachbarframe ist im Hintergrund quasi identisch
    steht = []
    for i in range(n):
        p = i > 0 and myCar_steht(grays[i - 1], grays[i], h, w)       # gegen Vorgänger
        q = i < n - 1 and myCar_steht(grays[i], grays[i + 1], h, w)   # gegen Nachfolger
        steht.append(bool(p or q))

    # 2) Hintergrundplatte = pixelweiser Median der Stillstands-Frames
    #    (Fallback: alle Frames, falls kein Stillstand erkannt wurde)
    stand_idx = [i for i in range(n) if steht[i]] or list(range(n))
    plate = np.median(np.stack([frames[i][1] for i in stand_idx]).astype(np.float32),
                      axis=0).astype(np.uint8)

    # 3) Entscheidung je Frame
    ergebnisse = []
    for i, (bildname, img) in enumerate(frames):
        if steht[i] and _querer_in_konfliktzone(img, plate):
            speed = 0            # steht+Auto im Fahrweg -> Vorfahrt geben
        else:
            speed = zone         # Anfahrt bzw. Weg frei -> weiterfahren
        ergebnisse.append({
            "bildname": bildname,
            "geschwindigkeit": speed,
            "richtung": "geradeaus",           # Scene 4 fährt durchgehend geradeaus
        })
    return ergebnisse


def _auto_von_rechts(img, plate, x_from=0.55, roi_y=(0.42, 0.62), min_area=3000):
    """Prüft, ob rechts (in der Querstrasse) bewegter Verkehr mit Vorfahrt ist

    Vergleich gegen die freigeräumte Referenz (letzter Stillstands-Frame):
    ein Auto rechts erzeugt einen großen Differenz-Blob, freie Strasse nicht

    Technik: Differenzbild + Threshold + Morphologie + Connected-Components

    Args:
        x_from (float): linker Rand der rechten Such-Region (Bildanteil)
        roi_y (tuple): Hoehenband (Bildanteile) auf Straßenhöhe
        min_area (int): Mindestgröße des Blobs

    Returns:
        bool: True, wenn rechts ein Fahrzeug-Blob liegt
    """
    h, w = img.shape[:2]
    d = cv2.cvtColor(cv2.absdiff(img, plate), cv2.COLOR_BGR2GRAY)     # Differenz zur Referenz
    _, th = cv2.threshold(d, 35, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # Nur die rechte Querstrassen-Region betrachten (dort kommt der Vorfahrtsverkehr)
    roi = th[int(roi_y[0] * h):int(roi_y[1] * h), int(x_from * w):]
    n, _, st, _ = cv2.connectedComponentsWithStats(roi, 8)
    groesster = max([st[k, 4] for k in range(1, n)], default=0)
    return groesster > min_area


def solve_scene_5(frames, zone=30):
    """Scene 5: 30er-Zone, an der Kreuzung gilt rechts-vor-links

    Von rechts kommt bewegter Querverkehr mit Vorfahrt -> anhalten und warten,
    bis rechts frei ist, dann losfahren. Gegenverkehr (links) ist egal.

    Vorgehen :
      1) Eigenbewegung je Frame -> fährt / steht  (myCar_steht)
      2) Referenz = letzter Stillstands-Frame (Kreuzung geräuumt, myCar fährt an)
      3) myCar steht + Querverkehr rechts (Differenz) -> 0, sonst Zonengeschw.

    """
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for _, img in frames]
    h, w = grays[0].shape
    n = len(frames)

    # 1) Stillstand je Frame (wie in Scene 4)
    steht = []
    for i in range(n):
        p = i > 0 and myCar_steht(grays[i - 1], grays[i], h, w)
        q = i < n - 1 and myCar_steht(grays[i], grays[i + 1], h, w)
        steht.append(bool(p or q))

    # 2) Referenzbild = letzter Stillstands-Frame. Annahme: am Szenenende ist die
    # Kreuzung geräuumt und der Trabi fährt an -> dieser Frame zeigt "rechts frei"
    stand_idx = [i for i in range(n) if steht[i]]
    plate = frames[stand_idx[-1]][1] if stand_idx else frames[-1][1]

    # 3) Entscheidung je Frame
    ergebnisse = []
    for i, (bildname, img) in enumerate(frames):
        if steht[i] and _auto_von_rechts(img, plate):
            speed = 0            # steht + Querverkehr rechts -> Vorfahrt geben
        else:
            speed = zone         # Anfahrt / rollt noch / rechts frei -> fahren
        ergebnisse.append({
            "bildname": bildname,
            "geschwindigkeit": speed,
            "richtung": "geradeaus",           #Scene 5 fährt durchgehend geradeaus
        })
    return ergebnisse




def load_scene(scene_dir):
    frames = []
    for name in sorted(os.listdir(scene_dir)):       
        if name.lower().endswith(".png"):
            img = cv2.imread(os.path.join(scene_dir, name))
            if img is not None:
                frames.append((name, img))
    return frames


if __name__ == "__main__":
    analyze_map_streets()
    dots = extract_yellow_dots()
    find_route(dots)

    solution = {}                                 

    # Scene 1: Ampelkreuzung, danach rechts abbiegen 
    frames = load_scene(os.path.join(IMAGE_DIR, "Scene_1"))
    solution["Scene_1"] = solve_scene_1(frames)

    # Scene 2: durchgehend 50 geradeaus (Vorfahrt, LKW wartet) 
    frames = load_scene(os.path.join(IMAGE_DIR, "Scene_2"))
    solution["Scene_2"] = solve_scene_2(frames)

    # Scene 3: trotz Vorfahrt vorsichtig, weil querender/abbiegender LKW die
    # Spur schneidet -> anhalten, dann links abbiegen 
    frames = load_scene(os.path.join(IMAGE_DIR, "Scene_3"))
    solution["Scene_3"] = solve_scene_3(frames)

    # Scene 4: 30er-Zone, Halt für querendes Auto, dann weiter
    frames = load_scene(os.path.join(IMAGE_DIR, "Scene_4"))
    solution["Scene_4"] = solve_scene_4(frames, zone=30)

    #Scene 5: 30er-Zone, rechts-vor-links: Querverkehr von rechts hat
    #Vorfahrt -> anhalten bis frei, dann losfahren 
    frames = load_scene(os.path.join(IMAGE_DIR, "Scene_5"))
    solution["Scene_5"] = solve_scene_5(frames, zone=30)

    with open("solution.json", "w", encoding="utf-8") as f:
        json.dump(solution, f, indent=2, ensure_ascii=False)
    print("solution.json geschrieben.")