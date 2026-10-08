import os
import json
import re

import torch
import torch.nn as nn
import torch.optim as optim

from model import KIModel


DATEN_ORDNER = "Daten"
MODEL_ORDNER = "models"
MODEL_DATEI = os.path.join(MODEL_ORDNER, "model_01.pt")


# --------------------------------------------------
# 1. JSON-Daten laden
# --------------------------------------------------

def daten_laden():
    daten = []

    if not os.path.exists(DATEN_ORDNER):
        print("Fehler: Der Ordner 'Daten' existiert nicht.")
        return daten

    for dateiname in os.listdir(DATEN_ORDNER):

        if not dateiname.lower().endswith(".json"):
            continue

        pfad = os.path.join(DATEN_ORDNER, dateiname)

        try:
            with open(pfad, "r", encoding="utf-8") as datei:
                inhalt = json.load(datei)

            if isinstance(inhalt, list):
                daten.extend(inhalt)
                print(f"Geladen: {dateiname} ({len(inhalt)} Einträge)")

        except Exception as fehler:
            print(f"Fehler beim Laden von {dateiname}: {fehler}")

    return daten


# --------------------------------------------------
# 2. Text in Wörter zerlegen
# --------------------------------------------------

def tokenisieren(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)


# --------------------------------------------------
# 3. Vokabular erstellen
# --------------------------------------------------

def vokabular_erstellen(daten):

    vokabular = {
        "<UNK>": 0
    }

    for eintrag in daten:

        frage = eintrag.get("frage", "")

        for token in tokenisieren(frage):

            if token not in vokabular:
                vokabular[token] = len(vokabular)

    return vokabular


# --------------------------------------------------
# 4. Frage in Zahlen umwandeln
# --------------------------------------------------

def frage_vektor(frage, vokabular):

    vektor = torch.zeros(len(vokabular))

    for token in tokenisieren(frage):

        nummer = vokabular.get(token, 0)

        vektor[nummer] += 1

    return vektor


# --------------------------------------------------
# 5. Antworten nummerieren
# --------------------------------------------------

def antworten_erstellen(daten):

    antworten = []

    for eintrag in daten:

        antwort = eintrag.get("antwort", "")

        if antwort not in antworten:
            antworten.append(antwort)

    return antworten


# --------------------------------------------------
# 6. Training
# --------------------------------------------------

def trainieren():

    print("")
    print("================================")
    print("      TRAINING STARTET")
    print("================================")
    print("")

    daten = daten_laden()

    if not daten:
        print("Keine Trainingsdaten gefunden.")
        return

    print("")
    print(f"Trainingsbeispiele: {len(daten)}")

    vokabular = vokabular_erstellen(daten)

    print(f"Vokabular: {len(vokabular)} Wörter")

    antworten = antworten_erstellen(daten)

    print(f"Antwortklassen: {len(antworten)}")

    # Fragen und Zielwerte vorbereiten
    X = []
    Y = []

    for eintrag in daten:

        frage = eintrag.get("frage", "")
        antwort = eintrag.get("antwort", "")

        if not frage or not antwort:
            continue

        X.append(frage_vektor(frage, vokabular))
        Y.append(antworten.index(antwort))

    X = torch.stack(X)
    Y = torch.tensor(Y, dtype=torch.long)

    print("")
    print("Daten vorbereitet.")
    print(f"Input-Größe: {X.shape[1]}")
    print(f"Output-Größe: {len(antworten)}")

    # --------------------------------------------------
    # Modell erstellen
    # --------------------------------------------------

    modell = KIModel(
        input_size=len(vokabular),
        hidden_size=128,
        output_size=len(antworten)
    )

    verlustfunktion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        modell.parameters(),
        lr=0.001
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    epochen = 500

    print("")
    print("Training...")
    print("")

    for epoche in range(epochen):

        optimizer.zero_grad()

        ausgabe = modell(X)

        verlust = verlustfunktion(
            ausgabe,
            Y
        )

        verlust.backward()

        optimizer.step()

        if (epoche + 1) % 25 == 0:

            vorhersagen = torch.argmax(
                ausgabe,
                dim=1
            )

            genauigkeit = (
                vorhersagen == Y
            ).float().mean().item() * 100

            print(
                f"Epoche {epoche + 1}/{epochen} | "
                f"Verlust: {verlust.item():.4f} | "
                f"Genauigkeit: {genauigkeit:.1f}%"
            )

    # --------------------------------------------------
    # Modell speichern
    # --------------------------------------------------

    os.makedirs(
        MODEL_ORDNER,
        exist_ok=True
    )

    speicherstand = {
        "model_state": modell.state_dict(),
        "vokabular": vokabular,
        "antworten": antworten
    }

    torch.save(
        speicherstand,
        MODEL_DATEI
    )

    print("")
    print("================================")
    print("       TRAINING FERTIG")
    print("================================")
    print("")
    print(f"Modell gespeichert:")
    print(MODEL_DATEI)
    print("")


if __name__ == "__main__":
    trainieren()
