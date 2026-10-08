import os
import json

DATEN_ORDNER = "Daten"


def daten_laden():
    alle_daten = []

    if not os.path.exists(DATEN_ORDNER):
        return alle_daten

    for dateiname in os.listdir(DATEN_ORDNER):
        if not dateiname.lower().endswith(".json"):
            continue

        pfad = os.path.join(DATEN_ORDNER, dateiname)

        try:
            with open(pfad, "r", encoding="utf-8") as datei:
                daten = json.load(datei)

            if isinstance(daten, list):
                alle_daten.extend(daten)

        except Exception as fehler:
            print(f"Fehler in {dateiname}: {fehler}")

    return alle_daten


def antwort_finden(frage):
    daten = daten_laden()

    for eintrag in daten:
        gespeicherte_frage = eintrag.get("frage", "")

        if frage.strip().lower() == gespeicherte_frage.strip().lower():
            return eintrag.get("antwort", "")

    return "Das habe ich noch nicht gelernt."
