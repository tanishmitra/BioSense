from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/health")
def health():
    return jsonify({"status": "running"})


@app.route("/alerts")
def alerts():
    return jsonify({
        "alerts": [
            {
                "patient": "patient_1",
                "type": "FALL_DETECTED"
            }
        ]
    })


if __name__ == "__main__":
    app.run(debug=True)