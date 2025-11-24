# hisse_simulasyon.py

import yfinance as yf
from datetime import datetime
import dateparser
from typing import Optional, Dict, Any

SIMULATION_FORMAT_HINT = "'AKBNK'a 6 ay önce 10.000 TL yatırsaydım ne olurdu?'"
SIMULATION_ERROR_MESSAGE = (
    "Simülasyon sırasında teknik bir hata oluştu. "
    "Lütfen soruyu şu formatta tekrar dener misin: "
    f"{SIMULATION_FORMAT_HINT}"
)

DEFAULT_TICKERS = ['KCHOL.IS', 'THYAO.IS', 'GARAN.IS', 'AKBNK.IS']
DEFAULT_SIMULATION_AMOUNT = 10000.0
DEFAULT_PERIOD_TEXT = "6 ay önce"


def build_simulation_error(reason: Optional[str] = None):
    """Kullanıcıya gösterilecek standart hata mesajını üret."""
    if reason:
        return {"hata": f"{reason}\n\n{SIMULATION_ERROR_MESSAGE}"}
    return {"hata": SIMULATION_ERROR_MESSAGE}


def _normalize_symbol(raw_symbol: Optional[str]) -> str:
    if not raw_symbol:
        return DEFAULT_TICKERS[0]

    symbol = raw_symbol.upper()
    if not symbol.endswith('.IS'):
        symbol = f"{symbol}.IS"

    # Hatalı uzunlukları engelle
    if len(symbol.replace('.IS', '')) < 2:
        return DEFAULT_TICKERS[0]
    return symbol


def _resolve_period_text(period_type: Optional[str], period_value: Optional[float], start_date: Optional[str]) -> str:
    if period_type == "date" and start_date:
        return start_date

    if period_type in {"months", "years", "days"} and period_value:
        value_int = int(round(period_value))
        value_int = max(1, value_int)
        unit_map = {
            "months": "ay",
            "years": "yıl",
            "days": "gün",
        }
        unit = unit_map.get(period_type, "ay")
        return f"{value_int} {unit} önce"

    return DEFAULT_PERIOD_TEXT


def _resolve_amount(amount_try: Optional[float]) -> float:
    if amount_try and amount_try > 0:
        return float(amount_try)
    return DEFAULT_SIMULATION_AMOUNT


def resolve_simulation_inputs(parsed_request: Dict[str, Any]) -> Dict[str, Any]:
    if parsed_request is None:
        parsed_request = {}

    symbol = _normalize_symbol(parsed_request.get("ticker"))
    amount = _resolve_amount(parsed_request.get("amount_try"))
    date_text = _resolve_period_text(
        parsed_request.get("period_type"),
        parsed_request.get("period_value"),
        parsed_request.get("start_date"),
    )

    return {
        "symbol": symbol,
        "amount": amount,
        "date_text": date_text,
    }


def run_simulation_from_parsed_request(parsed_request: Dict[str, Any]):
    """LLM'den dönen sözlükle simülasyonu çalıştır."""
    inputs = resolve_simulation_inputs(parsed_request)
    result = hisse_simulasyon(inputs["symbol"], inputs["date_text"], inputs["amount"])
    return result, inputs


def hisse_simulasyon(hisse_kodu: str, baslangic_input: str, yatirim_tutari: float):
    try:
        # 1. Hisse kodunu düzenle (Türk hisseleri için .IS ekle)
        if not hisse_kodu.endswith('.IS') and len(hisse_kodu) <= 6:
            hisse_kodu = f"{hisse_kodu}.IS"
        
        # 2. Doğal dil tarihini datetime objesine çevir
        baslangic_tarihi = dateparser.parse(baslangic_input)
        if not baslangic_tarihi:
            return build_simulation_error(f"Başlangıç tarihi anlaşılamadı: {baslangic_input}")

        baslangic_str = baslangic_tarihi.strftime("%Y-%m-%d")
        bugun = datetime.now().strftime("%Y-%m-%d")

        # 3. Veri çek - farklı formatları dene
        df = None
        symbol_variants = [hisse_kodu, hisse_kodu.replace('.IS', ''), f"{hisse_kodu.replace('.IS', '')}.IS"]
        
        for variant in symbol_variants:
            try:
                df = yf.download(variant, start=baslangic_str, end=bugun, progress=False)
                if not df.empty and len(df) >= 2:
                    break
            except:
                continue

        if df.empty or len(df) < 2:
            return build_simulation_error(f"{hisse_kodu} için yeterli veri bulunamadı.")

        # 3. İlk ve son fiyatı al
        ilk_gun_fiyati = df['Close'].iloc[0].item()
        son_fiyat = df['Close'].iloc[-1].item()

        # 4. Hesaplamalar
        lot_sayisi = yatirim_tutari / ilk_gun_fiyati
        simdiki_deger = lot_sayisi * son_fiyat
        kazanc = simdiki_deger - yatirim_tutari
        yuzde_getiri = (kazanc / yatirim_tutari) * 100

        return {
            "hisse": hisse_kodu,
            "başlangıç tarihi": baslangic_str,
            "başlangıç fiyatı": round(ilk_gun_fiyati, 2),
            "güncel fiyat": round(son_fiyat, 2),
            "alınan lot": round(lot_sayisi, 2),
            "şu anki değer": round(simdiki_deger, 2),
            "net kazanç": round(kazanc, 2),
            "getiri %": round(yuzde_getiri, 2)
        }

    except Exception as e:
        print(f"Simülasyon hatası: {e}")
        return build_simulation_error()


if __name__ == "__main__":
    print("📊 Hisse Senedi Simülasyon Aracı")
    print("-------------------------------------")

    hisse = input("Hisse kodunu girin (örn: THYAO.IS, ALARK.IS): ").strip().upper()
    tarih = input("Başlangıç tarihini girin (örn: 1 ay önce, 2023 başı, 2022-01-05): ").strip()
    tutar_input = input("Yatırım tutarı (TL): ").strip()

    try:
        tutar = float(tutar_input)
    except ValueError:
        print("❌ Geçersiz tutar!")
        exit()

    sonuc = hisse_simulasyon(hisse, tarih, tutar)

    print("\n📈 Simülasyon Sonucu:")
    for k, v in sonuc.items():
        print(f"{k}: {v}")
