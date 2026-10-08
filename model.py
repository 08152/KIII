import torch
import torch.nn as nn


class KIModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size=128,
        hidden_size=256
    ):
        super().__init__()

        # Wörter/Token → Zahlenvektoren
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # Verarbeitung der bisherigen Token
        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Hidden State → Wahrscheinlichkeit für jeden möglichen nächsten Token
        self.ausgabe = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x, hidden=None):

        # Token-IDs in Vektoren umwandeln
        x = self.embedding(x)

        # Sequenz verarbeiten
        x, hidden = self.gru(
            x,
            hidden
        )

        # Für jeden Token den nächsten Token vorhersagen
        x = self.ausgabe(x)

        return x, hidden
