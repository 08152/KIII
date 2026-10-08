import json
import os
from tokenizer import Tokenizer

DATEN_ORDNER = "Daten"
MODELL_DATEI = "vokabular.json"


def daten_laden():
    fragen = []
    antworten = []

    for dateiname in os.listdir(DATEN_ORDNER):
        if not dateiname.endswith(".json"):
            continue

        pfad = os.path.join(DATEN_ORDNER, dateiname)

        with open(pfad, "r", encoding="utf-8") as datei:
            daten = json.load(datei)

        for eintrag in daten:
            frage = eintrag.get("frage")
            antwort = eintrag.get("antwort")

            if frage and antwort:
                fragen.append(frage)
                antworten.append(antwort)

    return fragen, antworten


def trainieren():
    print("Lade Datensatz...")

    fragen, antworten = daten_laden()

    print(f"{len(fragen)} Datensätze gefunden.")

    tokenizer = Tokenizer()

    tokenizer.vokabular_erstellen(fragen)

    print(f"Vokabular: {len(tokenizer.vokabular)} Wörter/Zeichen")

    daten = {
        "vokabular": tokenizer.vokabular,
        "beispiele": []
    }

    for frage, antwort in zip(fragen, antworten):
        daten["beispiele"].append({
            "frage": tokenizer.encode(frage),
            "antwort": antwort
        })

    with open(MODELL_DATEI, "w", encoding="utf-8") as datei:
        json.dump(
            daten,
            datei,
            ensure_ascii=False,
            indent=2
        )

    print("Training abgeschlossen.")
    print(f"Gespeichert als: {MODELL_DATEI}")


if __name__ == "__main__":
    trainieren()
