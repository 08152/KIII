import re


class Tokenizer:

    def __init__(self):

        self.vokabular = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3
        }

        self.umgekehrt = {}

    def tokenisieren(self, text):

        return re.findall(
            r"\w+|[^\w\s]",
            text.lower(),
            re.UNICODE
        )

    def vokabular_erstellen(self, texte):

        for text in texte:

            tokens = self.tokenisieren(text)

            for token in tokens:

                if token not in self.vokabular:

                    self.vokabular[token] = len(
                        self.vokabular
                    )

        self.umgekehrt = {
            nummer: token
            for token, nummer
            in self.vokabular.items()
        }

    def encode(self, text):

        tokens = self.tokenisieren(text)

        return [
            self.vokabular.get(
                token,
                self.vokabular["<UNK>"]
            )
            for token in tokens
        ]

    def decode(self, zahlen):

        if not self.umgekehrt:

            self.umgekehrt = {
                nummer: token
                for token, nummer
                in self.vokabular.items()
            }

        tokens = []

        for zahl in zahlen:

            token = self.umgekehrt.get(
                int(zahl),
                "<UNK>"
            )

            if token in [
                "<PAD>",
                "<START>",
                "<END>"
            ]:
                continue

            tokens.append(token)

        text = " ".join(tokens)

        # Leerzeichen vor Satzzeichen entfernen
        text = re.sub(
            r"\s+([,.!?;:])",
            r"\1",
            text
        )

        return text
