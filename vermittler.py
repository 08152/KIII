```python
import os
import json
import re
import math

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


def woerter(text):
    return set(
        re.findall(r"\w+", text.lower(), re.UNICODE)
    )


def aehnlichkeit(frage1, frage2):
    woerter1 = woerter(frage1)
    woerter2 = woerter(frage2)

    if not woerter1 or not woerter2:
        return 0.0

    gemeinsame_woerter = woerter1 & woerter2

    gesamt = woerter1 | woerter2

    return len(gemeinsame_woerter) / len(gesamt)


def antwort_finden(frage):
    daten = daten_laden()

    if not daten:
        return "Ich habe noch keine Daten gelernt."

    beste_antwort = None
    bester_wert = 0.0

    for eintrag in daten:
        gespeicherte_frage = eintrag.get("frage", "")
        antwort = eintrag.get("antwort", "")

        wert = aehnlichkeit(frage, gespeicherte_frage)

        if wert > bester_wert:
            bester_wert = wert
            beste_antwort = antwort

    # Mindestähnlichkeit
    if bester_wert >= 0.25:
        return beste_antwort

    return "Das habe ich noch nicht gelernt."
```
