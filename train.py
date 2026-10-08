import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from model import KIModel
from tokenizer import Tokenizer


DATEN_ORDNER = "Daten"
MODEL_ORDNER = "models"
MODEL_DATEI = os.path.join(MODEL_ORDNER, "model_01.pt")


# ----------------------------------------
# Daten laden
# ----------------------------------------

def daten_laden():

    texte = []

    if not os.path.exists(DATEN_ORDNER):
        print("Daten-Ordner nicht gefunden.")
        return texte

    for dateiname in os.listdir(DATEN_ORDNER):

        if not dateiname.endswith(".json"):
            continue

        pfad = os.path.join(
            DATEN_ORDNER,
            dateiname
        )

        try:

            with open(
                pfad,
                "r",
                encoding="utf-8"
            ) as datei:

                daten = json.load(datei)

            if isinstance(daten, list):

                for eintrag in daten:

                    frage = eintrag.get(
                        "frage",
                        ""
                    )

                    antwort = eintrag.get(
                        "antwort",
                        ""
                    )

                    if frage and antwort:

                        # Frage und Antwort werden
                        # zu einem Trainingsbeispiel
                        text = (
                            "<START> "
                            + frage
                            + " "
                            + antwort
                            + " <END>"
                        )

                        texte.append(text)

                print(
                    f"{dateiname}: "
                    f"{len(daten)} Einträge"
                )

        except Exception as fehler:

            print(
                f"Fehler bei {dateiname}: "
                f"{fehler}"
            )

    return texte


# ----------------------------------------
# Training
# ----------------------------------------

def trainieren():

    print()
    print("==============================")
    print("     KI-TRAINING START")
    print("==============================")
    print()

    texte = daten_laden()

    if not texte:

        print("Keine Trainingsdaten gefunden.")
        return

    print()
    print(
        f"Trainingssätze: {len(texte)}"
    )

    # ------------------------------------
    # Tokenizer
    # ------------------------------------

    tokenizer = Tokenizer()

    tokenizer.vokabular_erstellen(
        texte
    )

    vocab_size = len(
        tokenizer.vokabular
    )

    print(
        f"Vokabular: {vocab_size} Token"
    )

    # ------------------------------------
    # Trainingsdaten erstellen
    # ------------------------------------

    X = []
    Y = []

    for text in texte:

        tokens = tokenizer.encode(text)

        if len(tokens) < 2:
            continue

        # Jeder Token soll den nächsten Token vorhersagen
        eingabe = tokens[:-1]
        ziel = tokens[1:]

        X.append(eingabe)
        Y.append(ziel)

    if not X:

        print("Keine gültigen Trainingsdaten.")
        return

    max_len = max(
        len(x)
        for x in X
    )

    pad_id = tokenizer.vokabular[
        "<PAD>"
    ]

    # Padding
    X_padded = []
    Y_padded = []

    for x, y in zip(X, Y):

        x = x + [
            pad_id
        ] * (
            max_len - len(x)
        )

        y = y + [
            pad_id
        ] * (
            max_len - len(y)
        )

        X_padded.append(x)
        Y_padded.append(y)

    X = torch.tensor(
        X_padded,
        dtype=torch.long
    )

    Y = torch.tensor(
        Y_padded,
        dtype=torch.long
    )

    print(
        f"Sequenzlänge: {max_len}"
    )

    # ------------------------------------
    # Modell
    # ------------------------------------

    modell = KIModel(
        vocab_size=vocab_size,
        embedding_size=128,
        hidden_size=256
    )

    verlustfunktion = nn.CrossEntropyLoss(
        ignore_index=pad_id
    )

    optimizer = optim.Adam(
        modell.parameters(),
        lr=0.001
    )

    # ------------------------------------
    # Training
    # ------------------------------------

    epochen = 500

    print()
    print("Training läuft...")
    print()

    for epoche in range(epochen):

        optimizer.zero_grad()

        ausgabe, _ = modell(X)

        verlust = verlustfunktion(
            ausgabe.reshape(
                -1,
                vocab_size
            ),
            Y.reshape(-1)
        )

        verlust.backward()

        optimizer.step()

        if (epoche + 1) % 25 == 0:

            print(
                f"Epoche "
                f"{epoche + 1}/{epochen} | "
                f"Verlust: "
                f"{verlust.item():.4f}"
            )

    # ------------------------------------
    # Modell speichern
    # ------------------------------------

    os.makedirs(
        MODEL_ORDNER,
        exist_ok=True
    )

    speicherstand = {

        "model_state": (
            modell.state_dict()
        ),

        "vokabular": (
            tokenizer.vokabular
        ),

        "config": {

            "vocab_size": vocab_size,

            "embedding_size": 128,

            "hidden_size": 256
        }
    }

    torch.save(
        speicherstand,
        MODEL_DATEI
    )

    print()
    print("==============================")
    print("       TRAINING FERTIG")
    print("==============================")
    print()
    print(
        f"Gespeichert: {MODEL_DATEI}"
    )


if __name__ == "__main__":

    trainieren()
