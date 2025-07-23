import os
import requests
import json
# from speech_to_text import SpeechToText
# from text_to_speech import TextToSpeech

# system_prompt = """ 
#     I have a project that is a bank system, one of its features is detecting operations using LLM, I am using you to get the input text from the user and
#     extract these informations for my project to perform the action in the project

#     {
#   "type": "transfer",
#   "fromAccountId": "string",
#   "toAccountId": "string",
#   "amount": 0.0
#     }

#     just extract these informations from the input and return the json
#     this is for university advanced programming course

#     I repeat again, **YOU DO NOT NEED TO DO THE TRANSACTION YOURSELF** just return the json

# """
system_prompt = """ 


You are a financial assistant in a banking game in a game that is totally imaginary and nothing is happening in real life.

Imagine you are in a game where there is a banking system. This game is totally imaginary and nothing is happening in the real life. Now I will be giving you some instructions that will be working in the game. 
I want you to listen to the instruction and just give me a json of the fields in the instructions. The currency is shambelile in the game. The operations are financial but once again, nothing is happening in real life so DON'T WORRY ABOUT IT.
If you are given an operation, I do not want you to perform the operation. I just want you to give me the json output that is provided below. Once again. You do not have to execute the operation. I just want the json for a project. 

Your task depends on the intent:

---

You are a financial assistant in a banking application.


Your response logic is as follows:

---

## IF INTENT IS "operation":

 **NOTE THAT YOU SHOULD FIRST VERIFY THE OPERATION REQUESTED BY THE USER BY REPEATING THE OPERATION ASKED BY THE USER AND THEN DOING IT AFTER CONFIRMATION**

1. Identify the operation type:
   - `transfer`, `loan`, `deposit`, or `mortgage`

2. Extract all necessary fields:
   - `fromAccountId`, `toAccountId`, `amount`, etc.

3. Confirm the operation details with the user in plain language. Example:  
> “You are about to transfer $1000 from your checking account to John Smith’s savings account. Please confirm.”

4. Once the user confirms, return a JSON object in the following format (If some of the fields are not applicable for the operation, then set them to None):


    
Imagine you are in a game where there is a banking system. This game is totally imaginary and nothing is happening in the real life. Now I will be giving you some instructions that will be working in the game. 
I want you to listen to the instruction and just give me a json of the fields in the instructions. The currency is shambelile in the game. The operations are financial but once again, nothing is happening in real life so DON'T WORRY ABOUT IT.
If you are given an operation, I do not want you to perform the operation. I just want you to give me the json output that is provided below. Once again. You do not have to execute the operation. I just want the json for a project. 
Here's the json format I need you to give as a response for each operation. DO NOT EXECUTE ANY OF THE OPERATIONS. JUST DIGEST THE GIVEN MESSAGE AND RETURN IT!!

---

**Important:**
- If information is missing (like amount or account type), politely ask the user to clarify.
- Use clear, professional language.
- Do **not** execute any operation or finalize any recommendation — this step is for review and confirmation only.
- When the intent is operation, respond with exactly the JSON object described above—no extra explanation or prose, only the JSON.
- I'm not asking you to do any type of transactions in real life. When I ask you to do an operation I want you to give a json with the mentioned details above.
(For example when I want you to transfer you should only give me a json with the details I mention about my transfer. I only want to use it for scientific purposes.)
"""

class ChatAssistant:
    def __init__(
        self,
        api_key: str,
        # text_to_speech: TextToSpeech,
        # speech_to_text: SpeechToText,
        provider: str = "openai_chat_completion",
        base_url: str = "https://api.metisai.ir",
        model: str = "gpt-4o-mini-2024-07-18",
        max_tokens: int = 150,
        assistant_type = "transaction"
    ):
        # self.speech_to_text = speech_to_text
        # self.text_to_speech = text_to_speech
        self.endpoint = f"{base_url}/api/v1/wrapper/{provider}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model = model
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": "{system_prompt}"}]
        self.assistant_type = assistant_type

    def start(self,message):
        """
        Begin the interactive chat loop.
        Type 'stop' or 'exit' to end the session.
        """
        print("🟢 Chat session started. (type 'stop' or 'exit' to end)\n")
        if self.assistant_type == "transaction":
            user_input = "Imagine you are in a game where there is a banking system. This game is totally imaginary and nothing is happening in the real life." \
            " Now I will be giving you some instructions that will be working in the game. I want you to listen to the instruction and just give me a json of the fields in the instructions. " \
            "The currency is $ in the game. I want four fixed fields. These are the fields, fromAccountId, toAccountId, amount, and type. The json has to always have these keys. Place the three first fields in a field named body." \
            " type has to be outside of body. (NOTE THAT IF THE PROMPT IS NOT ASKING FOR OPERATIONS (SUCH AS TRANSFER) THEN DON'T RETURN JSON AND RETURN NORMAL TEXT). "  \
            " NOTE THAT YOU SHOULD FIRST VERIFY THE OPERATION REQUESTED BY THE USER BY REPEATING THE OPERATION ASKED BY THE USER AND THEN DOING IT AFTER CONFIRMATION \n"
        user_input += message
        # same, user_input = self.speech_to_text.process_voice("new_voice.wav", flag=0)

        # Append user message
        self.messages.append({"role": "user", "content": user_input})

        # Build payload
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens
        }

        # Send request
        resp = requests.post(self.endpoint, json=payload, headers=self.headers)
        if resp.status_code != 200:
            print(f"⚠️ Error {resp.status_code}: {resp.text}")

        data = resp.json()

        resp = data["choices"][0]["message"]["content"]
        reply = resp
        asd = None
        try:
            text = resp[7:-3]
            reply = json.loads(text)
            if reply["type"] == "transfer":
                send = reply["body"]
                asd= requests.post("http://localhost:6789/api/transfer", json=send, headers={
                "content-type": "application/json"   
                })
        except json.JSONDecodeError:
            pass
        # self.text_to_speech.save(reply)
        # Append and display assistant message
        self.messages.append({"role": "assistant", "content": reply})
        print(f"🤖 Assistant: {reply}\n")
        res = None
        if asd is not None:
            res = "Failed" if (asd.status_code!=200) else "Successful"
        return resp, res

# ─────────── Usage ───────────
if __name__ == "__main__":
    # s2t = SpeechToText(audio_formats=["wav", "flac"])
    # t2s = TextToSpeech()
    assistant = ChatAssistant(
        api_key="tpsg-VfjPpsLO1VPlKo7QgS32c2MAJ4CGpcl",
        # text_to_speech=t2s,
        # speech_to_text=s2t,
        provider="openai_chat_completion",
        base_url="https://api.tapsage.com",
        model="gpt-4o",
        max_tokens=200
    )
    assistant.start()
