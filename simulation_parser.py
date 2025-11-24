import json
import re
from copy import deepcopy
from datetime import datetime

DEFAULT_PARSED_SIMULATION = {
    "is_simulation": False,
    "ticker": None,
    "amount_try": None,
    "period_type": None,
    "period_value": None,
    "start_date": None,
}

SIMULATION_KEYWORDS = [
    "simülasyon",
    "simulasyon",
    "simulation",
    "yatırım",
    "yatirim",
    "yatirsaydim",
    "yatırsaydım",
    "yatirsam",
    "yatırsa",
    "yatırsam",
    "ne olurdu",
    "kaç para",
    "kac para",
    "kazanç",
    "kazanc",
    "getiri",
]

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


def _looks_like_simulation(user_message: str) -> bool:
    if not user_message:
        return False
    msg = user_message.lower()
    return any(keyword in msg for keyword in SIMULATION_KEYWORDS)


def _resolve_year_reference(year: int, suffix: str) -> str:
    suffix = suffix.lower()
    if "son" in suffix:
        dt = datetime(year, 12, 1)
    else:
        dt = datetime(year, 1, 2)
    return dt.strftime("%Y-%m-%d")


def _fallback_parse_simulation(user_message: str) -> dict:
    parsed = deepcopy(DEFAULT_PARSED_SIMULATION)
    if not user_message:
        return parsed

    if not _looks_like_simulation(user_message):
        return parsed

    parsed["is_simulation"] = True

    # Ticker
    symbol_match = re.search(r'\b([A-ZÇĞİÖŞÜ]{2,6})\b', user_message.upper())
    if symbol_match:
        parsed["ticker"] = symbol_match.group(1)

    # Amount
    message_lower = user_message.lower()
    amount = None
    bin_match = re.search(r'(\d+(?:[.,]\d+)?)\s*bin', message_lower)
    if bin_match:
        amount = float(bin_match.group(1).replace(',', '.')) * 1000
    else:
        k_match = re.search(r'(\d+(?:[.,]\d+)?)\s*k\b', message_lower)
        if k_match:
            amount = float(k_match.group(1).replace(',', '.')) * 1000
    if amount is None:
        multi_dot = re.findall(r'\d+\.\d+\.\d+', user_message)
        if multi_dot:
            amount = max(int(num.replace('.', '')) for num in multi_dot)
    if amount is None:
        dot_numbers = re.findall(r'\d+\.\d+', user_message)
        if dot_numbers:
            amount = max(float(num.replace('.', '')) for num in dot_numbers)
    if amount is None:
        plain_numbers = re.findall(r'\d+', user_message)
        if plain_numbers:
            amount = max(int(num) for num in plain_numbers)
    if amount:
        parsed["amount_try"] = float(amount)

    # Period / date
    period_match = re.search(r'(\d+)\s*(ay|yıl|yil|hafta|gün|gun)\s*önce', message_lower)
    if period_match:
        value = float(period_match.group(1))
        unit = period_match.group(2)
        if unit in ['ay']:
            parsed["period_type"] = "months"
        elif unit in ['yıl', 'yil']:
            parsed["period_type"] = "years"
        elif unit in ['hafta']:
            parsed["period_type"] = "days"
            value *= 7
        else:
            parsed["period_type"] = "days"
        parsed["period_value"] = value
    else:
        date_match = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', user_message)
        if date_match:
            try:
                year, month, day = map(int, date_match.groups())
                parsed["period_type"] = "date"
                parsed["start_date"] = datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                pass
        else:
            year_phrase = re.search(r'(\d{4})\s*(başında|basinda|başı|basi|sonunda|sonu)', message_lower)
            if year_phrase:
                year = int(year_phrase.group(1))
                suffix = year_phrase.group(2)
                parsed["period_type"] = "date"
                parsed["start_date"] = _resolve_year_reference(year, suffix)

    return parsed


def parse_simulation_query(user_message: str, llm_model=None) -> dict:
    """
    LLM kullanarak simülasyon isteğini JSON formatında ayrıştır.
    Gemini/OpenAI gibi bir model yoksa kural tabanlı fallback devreye girer.
    """
    if not user_message:
        return deepcopy(DEFAULT_PARSED_SIMULATION)

    fallback_result = _fallback_parse_simulation(user_message)

    if not llm_model:
        return fallback_result

    prompt = build_simulation_parser_prompt(user_message)

    try:
        response = llm_model.generate_content(prompt)
        response_text = response.text.strip()
        json_block = _extract_json_block(response_text)
        if not json_block:
            return fallback_result
        payload = json.loads(json_block)
        normalized = _normalize_parsed_payload(payload)
        if not normalized["is_simulation"] and fallback_result["is_simulation"]:
            return fallback_result
        return normalized
    except Exception as exc:
        print(f"Simülasyon parser LLM hatası: {exc}")
        return fallback_result

