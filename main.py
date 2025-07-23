from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import uuid
import random
import io
import os

from bank_llm import ChatAssistant
from crypto_llm import api_crypto
from finance_llm import advise
from speech_to_text import SpeechToText
from text_to_speech import TextToSpeech
import config

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(os.getcwd(), "./")
os.makedirs(UPLOAD_DIR, exist_ok=True)

s2t = SpeechToText(audio_formats=["wav", "flac"])
t2s = TextToSpeech()
llm_agent = ChatAssistant(
        api_key=config.api_key,
        provider=config.provider,
        base_url=config.base_url,
        model=config.model,
        max_tokens=config.max_tokens
    )

def convert_to_wav(input_path: str) -> str:
    """
    Uses pydub/ffmpeg to convert any audio file to WAV.
    Returns the path to the new .wav file.
    """
    # load whatever format was uploaded
    audio = AudioSegment.from_file(input_path)
    wav_path = os.path.splitext(input_path)[0] + ".wav"
    # export as 16-bit PCM WAV (default)
    audio.export(wav_path, format="wav")
    return wav_path

@app.route("/llm/api/advise", methods=["POST"])
def respond_text():
    """
    Receives JSON body with a `text` field,
    processes it through your LLM, and returns JSON with `response`.
    """
    data = request.get_json(force=True)

    response_text = advise(data)

    return jsonify({"response": response_text})

@app.route("/llm/api/crypto", methods=["POST"])
def respond_text():
    data = request.get_json(force=True)
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    user_text = data['message']

    response_text, img_asset, img_corr = api_crypto(user_text)

    return jsonify({
        "response": response_text,
        "images": [
            f"data:image/png;base64,{img_asset}",
            f"data:image/png;base64,{img_corr}"
        ]
    })

@app.route("/llm/api/voice", methods=["POST"])
def respond_voice():
    """
    Receives an audio file (e.g. WAV/MP3) in the request,
    processes it, and returns a new audio response.
    """
    # 1) Retrieve uploaded voice file
    if 'audio' not in request.files:
        return jsonify({"error": "No file part"}), 400
    voice_file = request.files['audio']

    original_filename = secure_filename(voice_file.filename)
    saved_path = os.path.join(UPLOAD_DIR, original_filename)
    voice_file.save(saved_path)

    wav_path = convert_to_wav(saved_path)

    same, transcript = s2t.process_voice(wav_path, flag=1)
    response_text = "Security Protocol Failed. Initiating Self-Destruction Protocol in 5, 4, 3, 2, 1"
    res = ""
    if same is True:
        response_text, res = llm_agent.start(transcript)
        if res is None:
            res = ""
        elif res is not None:
            response_text = ""
    print(response_text)
    response_audio = t2s.save(response_text + "  " +  res)

    with open(response_audio, 'rb') as f:
        audio_bytes = f.read()
    

    return send_file(
        io.BytesIO(audio_bytes),
        mimetype=voice_file.mimetype,
        as_attachment=False,
    )

@app.route("/llm/api/text", methods=["POST"])
def respond_text():
    """
    Receives JSON body with a `text` field,
    processes it through your LLM, and returns JSON with `response`.
    """
    data = request.get_json(force=True)
    print(data)
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    user_text = data['message']
    response_text, res = llm_agent.start(user_text)

    if res is None:
        res = ""
    elif res is not None:
        response_text = ""
    return jsonify({"response": response_text + "  " + res})


# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(host="0.0.0.0" , port=5000, debug=True)
