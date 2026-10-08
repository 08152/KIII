import re


class Tokenizer:
    def __init__(self):
        self.vokabular = {
            "<PAD>": 0,
            "<UNK>": 1
        }

    def tokenisieren(self, text):
        text = text.lower()

        # Wörter und Satzzeichen getrennt erkennen
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

        return tokens

    def vokabular_erstellen(self, texte):
        for text in texte:
            tokens = self.tokenisieren(text)

            for token in tokens:
                if token not in self.vokabular:
                    self.vokabular[token] = len(self.vokabular)

    def encode(self, text):
        tokens = self.tokenisieren(text)

        return [
            self.vokabular.get(token, self.vokabular["<UNK>"])
            for token in tokens
        ]

    def decode(self, zahlen):
        umgekehrt = {
            nummer: token
            for token, nummer in self.vokabular.items()
        }

        return [
            umgekehrt.get(zahl, "<UNK>")
            for zahl in zahlen
        ]
