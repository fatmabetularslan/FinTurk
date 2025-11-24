import json
from copy import deepcopy

DEFAULT_PARSED_SIMULATION = {
    "is_simulation": False,
    "ticker": None,
    "amount_try": None,
    "period_type": None,
    "period_value": None,
    "start_date": None,
}

PARSER_PROMPT_TEMPLATE = """
Sen bir "Borsa Yatırım Simülasyonu Parser"ısın.

Görev:
- Kullanıcının Türkçe serbest metinle yazdığı yatırım sorusunu oku.
- Sadece aşağıdaki alanları içeren JSON üret:
  {{
    "is_simulation": true/false,
    "ticker": null veya "AKBNK",
    "amount_try": null veya sayı,
    "period_type": "months" | "years" | "days" | "date",
    "period_value": sayı veya null,
    "start_date": "YYYY-MM-DD" veya null
  }}

Kurallar:
- Simülasyon sorusu değilse is_simulation=false yap ve diğer tüm alanları null bırak.
- Kullanıcı "6 ay önce" derse period_type="months", period_value=6.
- "1 yıl önce" derse period_type="years", period_value=1.
- "2023-01-01'de" derse period_type="date", start_date="2023-01-01".
- "10.000 TL" gibi tutarları amount_try=10000 şeklinde sayı yap.
- Ticker kodu BIST kısaltmasıdır (AKBNK, THYAO, ASELS, vb).
- Sadece geçerli JSON döndür, başka açıklama yazma.

Örnekler:
Kullanıcı: "AKBNK'a 6 ay önce 10.000 TL yatırsaydım ne olurdu?"
Cevap:
{{
  "is_simulation": true,
  "ticker": "AKBNK",
  "amount_try": 10000,
  "period_type": "months",
  "period_value": 6,
  "start_date": null
}}

Kullanıcı: "2024'ün başında ASELS'e 20 bin yatırsam şimdi ne eder?"
Cevap:
{{
  "is_simulation": true,
  "ticker": "ASELS",
  "amount_try": 20000,
  "period_type": "date",
  "period_value": null,
  "start_date": "2024-01-02"
}}

Şimdi şu soruyu işle:
"{user_message}"
"""


def build_simulation_parser_prompt(user_message: str) -> str:
    if not user_message:
        user_message = ""
    return PARSER_PROMPT_TEMPLATE.format(user_message=user_message.replace('"', '\\"'))


def _extract_json_block(raw_text: str) -> str | None:
    if not raw_text:
        return None
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw_text[start : end + 1]


def _normalize_parsed_payload(payload: dict) -> dict:
    normalized = deepcopy(DEFAULT_PARSED_SIMULATION)
    if not isinstance(payload, dict):
        return normalized

    normalized["is_simulation"] = bool(payload.get("is_simulation"))

    ticker = payload.get("ticker")
    normalized["ticker"] = ticker.upper() if isinstance(ticker, str) else None

    amount = payload.get("amount_try")
    if isinstance(amount, (int, float)):
        normalized["amount_try"] = float(amount)
    elif isinstance(amount, str):
        try:
            normalized["amount_try"] = float(amount.replace(".", "").replace(",", "."))
        except ValueError:
            normalized["amount_try"] = None

    period_type = payload.get("period_type")
    normalized["period_type"] = period_type if period_type in {"months", "years", "days", "date"} else None

    period_value = payload.get("period_value")
    if isinstance(period_value, (int, float)):
        normalized["period_value"] = float(period_value)
    elif isinstance(period_value, str):
        try:
            normalized["period_value"] = float(period_value)
        except ValueError:
            normalized["period_value"] = None

    start_date = payload.get("start_date")
    normalized["start_date"] = start_date if isinstance(start_date, str) and start_date else None

    return normalized


def parse_simulation_query(user_message: str, llm_model=None) -> dict:
    """
    LLM kullanarak simülasyon isteğini JSON formatında ayrıştır.
    """
    if not user_message or not llm_model:
        return deepcopy(DEFAULT_PARSED_SIMULATION)

    prompt = build_simulation_parser_prompt(user_message)

    try:
        response = llm_model.generate_content(prompt)
        response_text = response.text.strip()
        json_block = _extract_json_block(response_text)
        if not json_block:
            return deepcopy(DEFAULT_PARSED_SIMULATION)
        payload = json.loads(json_block)
        return _normalize_parsed_payload(payload)
    except Exception as exc:
        print(f"Simülasyon parser LLM hatası: {exc}")
        return deepcopy(DEFAULT_PARSED_SIMULATION)

