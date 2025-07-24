# LLM Agents: Finance Assistant

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)]()

A Flask‑based **LLM‑powered Finance Assistant** demo built for the “LLM Agents” event.  
This product combines text and voice interfaces to:

1. **Execute secure bank transfers**  
2. **Offer personalized risk‑based financial advice**  
3. **Analyze and advise on the cryptocurrency market**  

---

## 🧩 Features

### 1. Bank Actions via Text & Voice  
- **Transfer funds** using simple text commands (e.g. “Transfer $200 to account 1234.”).  
- **Voice recognition**: Upload or speak an audio file; system converts speech → text.  
- **Fail‑safe & confirmation**: Verifies identity, confirms every transaction step.

### 2. Personalized Financial Advice  
- Analyzes user’s financial history.  
- Provides **risk‑profiled recommendations** on savings, investments, and portfolio allocation.

### 3. Crypto Market Analysis & “Get Rich” Plan  
- Fetches real‑time & historical crypto data (via CryptoCompare).  
- Performs quantitative analyses (moving averages, RSI, volatility, drawdowns).  
- Auto‑builds a **7‑day forecast** with Prophet.  
- Generates a **custom trading plan** to help users “get rich” responsibly.

---

## 🚀 Tech Stack

- **Backend Framework**: Flask + CORS  
- **LLM Integration**: Custom `ChatAssistant` wrapper over your chosen LLM API  
- **Speech**:  
  - `SpeechToText` (via Pydub + ffmpeg)  
  - `TextToSpeech` (audio generation)  
- **Data & Analytics**:  
  - Crypto data: CryptoCompare API  
  - FX rates: Frankfurter API  
  - Equities: Yahoo! Finance (yfinance)  
  - Time‑series forecasting: Prophet  
  - Analysis: pandas, numpy, matplotlib  
- **Deployment**: Containerize with Docker (optional)

---

## 🛠️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-org/llm-agents-finance-assistant.git
   cd llm-agents-finance-assistant
   ```

2. **Create & activate a virtualenv**

```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **FFmpeg (for audio conversion)**

```bash
# macOS (with Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
```

--- 

## 🔧 Configuration

Copy `config.example.py` → `config.py` and fill in:

```python
api_key    = "YOUR_LLM_API_KEY"
provider   = "openai_chat_completion"       # or your provider alias
base_url   = "https://api.metisai.ir"       # your LLM endpoint
model      = "gpt-4o-mini-2024-07-18"       # model name
max_tokens = 150
```

---

## 🚨 API Reference

All endpoints are rooted at:

```
http://<HOST>:5000/llm/api
```

---

## 📡 Endpoints

| Endpoint   | Method | Payload                            | Returns                         |
|------------|--------|------------------------------------|----------------------------------|
| `/text`    | POST   | `{ message: "<user text>" }`       | `{ response: "<assistant reply>" }` |
| `/voice`   | POST   | `form‑data: audio file (wav/mp3)`  | Audio reply stream               |
| `/advise`  | POST   | `{ text: "<user text>" }`          | `{ response: "<financial advice>" }` |
| `/crypto`  | POST   | `{ message: "<crypto question>" }` | `{ response: "...", images: [asset, corr_chart] }` |

---

## 🔒 Security & Fail‑Safe

- Identity check on every voice request; if it fails, triggers a “self‑destruct” placeholder.
- Explicit confirmations before executing any bank action.
- No hard‑coded secrets: All keys live in `config.py` or environment variables.

---

## 📂 Core Code Samples

### `app.py`

```python
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import io, os

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
    audio = AudioSegment.from_file(input_path)
    wav_path = os.path.splitext(input_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

@app.route("/llm/api/text", methods=["POST"])
def respond_text():
    data = request.get_json(force=True)
    if "message" not in data:
        return jsonify({error: "Missing 'message'"}), 400
    resp_text, res = llm_agent.start(data["message"])
    return jsonify({response: resp_text + "  " + (res or "")})

# ... other routes for /voice, /advise, /crypto ...

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

### `finance_llm.py`

```python
import json, requests, config

class ChatAssistant:
    def __init__(self, type, api_key, provider, base_url, model, max_tokens):
        self.type = type
        self.endpoint = f"{base_url}/api/v1/wrapper/{provider}/chat/completions"
        self.headers = {Authorization: f"Bearer {api_key}", Content-Type: "application/json"}
        self.model = model
        self.max_tokens = max_tokens
        self.messages = [{role: "system", content: system_prompt[type]}]

    def start(self, logs):
        if self.type in [deposit, withdraw, loan, transfer]:
            filtered = [tx for tx in logs if tx.get(type) == self.type]
            user_input = json.dumps(filtered)
        else:
            user_input = logs
        self.messages.append({role: "user", content: user_input})
        payload = {model: self.model, messages: self.messages, max_tokens: self.max_tokens}
        resp = requests.post(self.endpoint, json=payload, headers=self.headers)
        return resp.json()[choices][0][message][content]

def advise(message):
    with open("log.json", "r") as f:
        transactions = json.load(f)[transactions]
    analyses = {
        t: ChatAssistant(t, config.api_key, config.provider, config.base_url, config.model, config.max_tokens).start(transactions)
        for t in [deposit, withdraw, loan, transfer]
    }
    combined = "\n\n".join(f"--- {t.upper()} ---\n{analyses[t]}" for t in analyses)
    meta = ChatAssistant(meta-agent, config.api_key, config.provider, config.base_url, config.model, config.max_tokens)
    analyses[recommendation] = meta.start(combined)
    return "\n".join([f"# {k.capitalize()}\n{v}" for k, v in analyses.items()])
```

---

### 🤝 Contributing

Fork & clone the repository.

Create a feature branch:  
`git checkout -b feat/my-feature`

Commit with clear messages.

Open a Pull Request – we’ll review & merge!

---
### Authors   
-[Mahdi Alinejad](https://github.com/soilorian) 
-[Alireza Mirrokni](https://github.com/alirezamirrokni)  
-[Arshia Izadyari](https://github.com/arshiaizd) 

---

### 📄 License

This project is MIT‑licensed. See LICENSE for details.
