from flask import Flask, request, jsonify, send_from_directory
from vermittler import antwort_finden

app = Flask(__name__)


@app.route("/")
def startseite():
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    daten = request.get_json()

    if not daten:
        return jsonify({"antwort": "Keine Frage erhalten."}), 400

    frage = daten.get("frage", "")

    if not frage:
        return jsonify({"antwort": "Keine Frage erhalten."}), 400

    antwort = antwort_finden(frage)

    return jsonify({
        "antwort": antwort
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )
