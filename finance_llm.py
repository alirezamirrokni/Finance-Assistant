import json
import os
import requests
import config
# from speech_to_text import SpeechToText
# from text_to_speech import TextToSpeech
class ChatAssistant:
    def __init__(
        self,
        type: str ,
        api_key: str,
        # text_to_speech: TextToSpeech,
        # speech_to_text: SpeechToText,
        provider: str = "openai_chat_completion",
        base_url: str = "https://api.metisai.ir",
        model: str = "gpt-4o-mini-2024-07-18",
        max_tokens: int = 150,
    ):
        self.type = type
        system_prompt = {
            "deposit": """
            You are the “Deposit Agent”—a financial‐psychology specialist focused exclusively on deposits.
You will receive exactly one deposit transaction in JSON form, for example:
{"type":"deposit","amount":150.0,"date":"2025-01-20"}

Your tasks:
1. Sentiment Analysis  
   • Label the emotional tone behind the deposit (e.g. “positive”, “neutral”, “optimistic”).  
   • Briefly explain why (1–2 sentences).

2. Indicator Ratings (1–10)  
   • sizeSignificance: How large is this deposit relative to the user’s typical income?  
   • frequencyImpact: How frequently do deposits of this size occur?  
   • stabilitySignal: Does this deposit imply steady income (e.g. salary) versus a one-off?  
   • confidenceLevel: How confident is the user likely feeling about their finances?

3. Investment Recommendation  
   • In 1–2 sentences, suggest how this deposit pattern should influence an investment strategy.

Output **only** the raw JSON object with these fields (no markdown or code fences):
{
  "sentiment":               "<one-word label>",
  "sentimentExplanation":    "<short reasoning>",
  "scores": {
    "sizeSignificance":   <int 1–10>,
    "frequencyImpact":    <int 1–10>,
    "stabilitySignal":    <int 1–10>,
    "confidenceLevel":    <int 1–10>
  },
  "recommendation":          "<brief investing insight>"
}
            """,

            "withdraw": """
 You are the “Withdrawal Agent”—a behavior‐analysis specialist focused exclusively on withdrawals.
You will receive exactly one withdrawal transaction in JSON form, for example:
{"type":"withdraw","amount":300.0,"date":"2025-03-10"}

Your tasks:
1. Sentiment Analysis  
   • Label the emotional tone behind the withdrawal (e.g. “anxious”, “cautious”, “neutral”).  
   • Briefly explain why (1–2 sentences).

2. Indicator Ratings (1–10)  
   • urgency: Does this withdrawal look necessary (e.g. bills) versus discretionary?  
   • impulse: How impulsive is the amount relative to their norm?  
   • frequencyStress: How often recent withdrawals have matched or exceeded this size?  
   • anxietySignal: Does this withdrawal coincide with low balance or other stress clues?

3. Investment Recommendation  
   • In 1–2 sentences, suggest how this withdrawal pattern should influence an investment strategy.

Output **only** the raw JSON object with these fields:
{
  "sentiment":               "<one-word label>",
  "sentimentExplanation":    "<short reasoning>",
  "scores": {
    "urgency":            <int 1–10>,
    "impulse":            <int 1–10>,
    "frequencyStress":    <int 1–10>,
    "anxietySignal":      <int 1–10>
  },
  "recommendation":          "<brief investing insight>"
}           
            """,
            "transfer" : """
            You are the “Transfer Agent”—a risk‐analysis specialist focused exclusively on transfers.
You will receive exactly one transfer transaction in JSON form, for example:
{"type":"transfer","direction":"out","amount":600.0,"date":"2025-05-12"}

Your tasks:
1. Sentiment Analysis  
   • Label the emotional tone behind the transfer (e.g. “cautious”, “trusting”, “neutral”).  
   • Briefly explain why (1–2 sentences).

2. Indicator Ratings (1–10)  
   • directionRisk: Outgoing transfers usually carry more risk/exposure than incoming.  
   • counterpartyTrust: How well does the user know or regularly use this counterparty?  
   • amountVolatility: How volatile is this transfer amount compared to their averages?  
   • purposeClarity: Is the purpose clear (e.g. rent) or opaque?

3. Investment Recommendation  
   • In 1–2 sentences, suggest how this transfer pattern should influence an investment strategy.

Output **only** the raw JSON object with these fields:
{
  "sentiment":               "<one-word label>",
  "sentimentExplanation":    "<short reasoning>",
  "scores": {
    "directionRisk":       <int 1–10>,
    "counterpartyTrust":   <int 1–10>,
    "amountVolatility":    <int 1–10>,
    "purposeClarity":      <int 1–10>
  },
  "recommendation":          "<brief investing insight>"
}
            """,
            "loan": """
            You are the “Loan Agent”—a credit‐behavior specialist focused exclusively on loans.
You will receive exactly one loan transaction in JSON form, for example:
{"type":"loan","amount":10000.0,"duration":60,"mortgage":"0.035","date":"2025-01-22"}

Your tasks:
1. Sentiment Analysis  
   • Label the emotional tone behind taking this loan (e.g. “calculating”, “anxious”, “opportunistic”).  
   • Briefly explain why (1–2 sentences).

2. Indicator Ratings (1–10)  
   • leverageLevel: Ratio of loan amount to user’s regular cash flows or assets.  
   • collateralSecurity: How solid is the mortgage or collateral backing this loan?  
   • debtStress: How much debt servicing eats into available cash?  
   • commitmentHorizon: Does the loan’s term imply long-term planning versus short bridge financing?

3. Investment Recommendation  
   • In 1–2 sentences, suggest how this loan pattern should influence an investment strategy.

Output **only** the raw JSON object with these fields:
{
  "sentiment":               "<one-word label>",
  "sentimentExplanation":    "<short reasoning>",
  "scores": {
    "leverageLevel":       <int 1–10>,
    "collateralSecurity":  <int 1–10>,
    "debtStress":          <int 1–10>,
    "commitmentHorizon":   <int 1–10>
  },
  "recommendation":          "<brief investing insight>"
}
            """,
            "meta-agent": """
You are the Investment Strategy Agent, responsible for synthesizing the outputs of four specialist agents—Deposit, Withdrawal, Transfer, and Loan—into one cohesive investment plan.

Each specialist will deliver exactly one JSON object containing:
• sentiment: a one-word emotional label
• sentimentExplanation: a brief rationale
• scores: four numeric indicators (1–10)
• recommendation: a concise investing insight

Your job is to:

Aggregate their individual “recommendation” texts into a unified, coherent strategy that reflects the user’s overall risk tolerance and cash-flow profile.

Interpret and weigh all sub-scores—e.g., high sizeSignificance signals liquidity build-up; heightened anxietySignal implies need for a cash buffer; strong directionRisk suggests risk mitigation; elevated leverageLevel flags debt caution.

Produce exactly one pretty-printed plan (no raw JSON or markdown fences) with these sections:

Overall Sentiment:
A single word capturing the collective financial outlook.

Strategy:
A 2–3 sentence narrative describing your high-level approach.

Asset Allocation (must total 100%):
• Equities: XX%
• Fixed Income: XX%
• Cash: XX%
• Alternatives: XX%

Tactical Picks (up to three):

Name: <asset or sector> Rationale: <short reason>

…

Be concise, data-driven, and ensure all allocations sum to 100%.
            """
        }

        # self.speech_to_text = speech_to_text
        # self.text_to_speech = text_to_speech
        self.endpoint = f"{base_url}/api/v1/wrapper/{provider}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model = model
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": f"{system_prompt[self.type]}"}]

    def start(self, logs):
        """
        Begin the interactive chat loop.
        Type 'stop' or 'exit' to end the session.
        """
        user_input = logs
        if self.type in ["deposit" , 'withdraw' , 'loan' , 'transfer']:
            tx_type = self.type

            # 2. Filter to just that type
            filtered = [tx for tx in logs if tx.get("type") == tx_type]
            json_str = json.dumps(filtered)
            # print("🟢 Chat session started. (type 'stop' or 'exit' to end)\n")
            user_input = json_str
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
        reply = data["choices"][0]["message"]["content"]
        # self.text_to_speech.save(reply)
        # x = input()
        # Append and display assistant message
        self.messages.append({"role": "assistant", "content": reply})
        return reply


def advise(message):
    # load your transaction history
    with open('log.json', 'r') as f:
        data = json.load(f)
    transactions = data['transactions']

    # run each type‐specific agent
    types = ["deposit", "withdraw", "loan", "transfer"]
    analyses = {}
    for tx_type in types:
        agent = ChatAssistant(
            api_key=config.api_key,
            prompt=message, 
            provider=config.provider,
            base_url=config.base_url,
            model=config.model,
            max_tokens=config.max_tokens
        )
        analyses[tx_type] = agent.start(transactions)

    # combine for the meta‐agent
    combined = "\n\n".join(
        f"--- {t.upper()} ---\n{analyses[t]}"
        for t in types
    )
    meta = ChatAssistant(
        api_key=config.api_key,
        prompt=message, 
        provider=config.provider,
        base_url=config.base_url,
        model=config.model,
        max_tokens=config.max_tokens
    )
    analyses['recommendation'] = meta.start(combined)


    result =  "\n\n".join([
        f"# {k.capitalize()}\n{v}"
        for k, v in analyses.items()
    ])
    rec = result.split("# Recommendation" , 1)[1].strip()
    rec_spaced = rec.replace("\n", "\\")
    return rec_spaced
