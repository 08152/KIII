import os
import json
import re
import torch
import torch.nn as nn
from models.model_01 import Model01

DATEN_ORDNER = "Daten"
MODELS_ORDNER = "models"
MODELL_DATEI = os.path.join(MODELS_ORDNER, "model_01.pt")


def text_zu_woertern(text):
    return re.findall(r"\w+", text.lower(), re.UNICODE)


def daten_laden():
    daten = []

    for dateiname in os.listdir(DATEN_ORDNER):
        if not dateiname.endswith(".json"):
            continue

        pfad = os.path.join(DATEN_ORDNER, dateiname)

        with open(pfad, "r", encoding="utf-8") as datei:
            inhalt = json.load(datei)

        if isinstance(inhalt, list):
            daten.extend(inhalt)

    return daten


def trainieren():
    daten = daten_laden()

    if not daten:
        print("Keine Trainingsdaten gefunden.")
        return

    # Vokabular erstellen
    vokabular = {}

    for eintrag in daten:
        for wort in text_zu_woertern(eintrag["frage"]):
            if wort not in vokabular:
                vokabular[wort] = len(vokabular)

    # Antworten bekommen IDs
    antworten = []

    for eintrag in daten:
        if eintrag["antwort"] not in antworten:
            antworten.append(eintrag["antwort"])

    X = []
    y = []

    for eintrag in daten:
        vektor = [0.0] * len(vokabular)

        for wort in text_zu_woertern(eintrag["frage"]):
            if wort in vokabular:
                vektor[vokabular[wort]] = 1.0

        X.append(vektor)
        y.append(antworten.index(eintrag["antwort"]))

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    modell = Model01(
        input_size=len(vokabular),
        hidden_size=128,
        output_size=len(antworten)
    )

    verlustfunktion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        modell.parameters(),
        lr=0.001
    )

    print("Training gestartet...")

    for epoche in range(300):
        optimizer.zero_grad()

        ausgabe = modell(X)

        verlust = verlustfunktion(ausgabe, y)

        verlust.backward()
        optimizer.step()

        if (epoche + 1) % 50 == 0:
            print(
                f"Epoche {epoche + 1}/300 "
                f"| Verlust: {verlust.item():.4f}"
            )

    os.makedirs(MODELS_ORDNER, exist_ok=True)

    torch.save({
        "model_state": modell.state_dict(),
        "vokabular": vokabular,
        "antworten": antworten
    }, MODELL_DATEI)

    print()
    print("Training abgeschlossen!")
    print(f"Modell gespeichert: {MODELL_DATEI}")


if __name__ == "__main__":
    trainieren()
