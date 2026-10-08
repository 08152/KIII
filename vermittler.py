import os
import torch

from model import KIModel
from tokenizer import Tokenizer


MODEL_DATEI = "models/model_01.pt"

modell = None
tokenizer = None


def modell_laden():

    global modell
    global tokenizer

    if not os.path.exists(MODEL_DATEI):
        print("Kein trainiertes Modell gefunden.")
        return False

    daten = torch.load(
        MODEL_DATEI,
        map_location="cpu"
    )

    tokenizer = Tokenizer()
    tokenizer.vokabular = daten["vokabular"]

    tokenizer.umgekehrt = {
        nummer: token
        for token, nummer
        in tokenizer.vokabular.items()
    }

    config = daten["config"]

    modell = KIModel(
        vocab_size=config["vocab_size"],
        embedding_size=config["embedding_size"],
        hidden_size=config["hidden_size"]
    )

    modell.load_state_dict(
        daten["model_state"]
    )

    modell.eval()

    print("Modell erfolgreich geladen.")

    return True


def antwort_generieren(
    frage,
    max_tokens=80,
    temperatur=0.8
):

    if modell is None:

        if not modell_laden():
            return "Mein Modell wurde noch nicht trainiert."

    start_id = tokenizer.vokabular.get(
        "<START>",
        2
    )

    end_id = tokenizer.vokabular.get(
        "<END>",
        3
    )

    tokens = tokenizer.encode(frage)

    eingabe = [start_id] + tokens

    hidden = None

    erzeugte_tokens = []

    with torch.no_grad():

        for _ in range(max_tokens):

            x = torch.tensor(
                [eingabe],
                dtype=torch.long
            )

            ausgabe, hidden = modell(
                x,
                hidden
            )

            logits = ausgabe[0, -1]

            logits = logits / temperatur

            wahrscheinlichkeiten = torch.softmax(
                logits,
                dim=-1
            )

            naechster_token = torch.multinomial(
                wahrscheinlichkeiten,
                1
            ).item()

            if naechster_token == end_id:
                break

            erzeugte_tokens.append(
                naechster_token
            )

            eingabe = [naechster_token]

    antwort = tokenizer.decode(
        erzeugte_tokens
    )

    if not antwort.strip():
        return "Ich weiß noch nicht, was ich darauf antworten soll."

    return antwort


def antwort_finden(frage):

    return antwort_generieren(frage)
