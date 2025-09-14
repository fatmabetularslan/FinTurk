import streamlit as st
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from finta import TA
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from document_rag_agent import DocumentRAGAgent
from technical_analysis import TechnicalAnalysisEngine
from financial_calendar import FinancialCalendar
import uuid
import requests
from textblob import TextBlob
import re
from bs4 import BeautifulSoup
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'))
    print(f"Gemini API anahtarı yüklendi: {GEMINI_API_KEY[:10]}...")
else:
    print("Gemini API anahtarı bulunamadı. .env dosyasında GOOGLE_API_KEY veya GEMINI_API_KEY tanımlayın.")
    gemini_model = None

# News API Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '67b1d8b38f8b4ba8ba13fada3b9deac1')
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Initialize Document RAG Agent
try:
    document_rag_agent = DocumentRAGAgent()
    print("Document RAG Agent basariyla yuklendi")
except Exception as e:
    print(f"Document RAG Agent yuklenemedi: {e}")
    document_rag_agent = None

# Initialize Technical Analysis Engine
try:
    technical_analysis_engine = TechnicalAnalysisEngine()
    print("Technical Analysis Engine basariyla yuklendi")
except Exception as e:
    print(f"Technical Analysis Engine yuklenemedi: {e}")
    technical_analysis_engine = None

# Initialize Financial Q&A Agent
try:
    from financial_qa_agent import FinancialQAAgent
    financial_qa_agent = FinancialQAAgent()
    print("Financial Q&A Agent basariyla yuklendi")
except Exception as e:
    print(f"Financial Q&A Agent yuklenemedi: {e}")
    financial_qa_agent = None

# Initialize Investment Advisor
try:
    from investment_advisor import InvestmentAdvisor
    investment_advisor = InvestmentAdvisor()
    print("Investment Advisor başarıyla yüklendi")
except Exception as e:
    print(f"Investment Advisor yüklenemedi: {e}")
    investment_advisor = None

# Hisse simülasyon modülünü import et
try:
    from hisse_simulasyon import hisse_simulasyon
    print("Hisse Simülasyon modülü başarıyla yüklendi")
except Exception as e:
    print(f"Hisse Simülasyon modülü yüklenemedi: {e}")
    hisse_simulasyon = None

# Initialize Portfolio Manager
try:
    from portfolio_manager import PortfolioManager
    portfolio_manager = PortfolioManager()
    print("Portfolio Manager başarıyla yüklendi")
except Exception as e:
    print(f"Portfolio Manager yüklenemedi: {e}")
    portfolio_manager = None

# Initialize Financial Calendar
try:
    financial_calendar = FinancialCalendar()
    print("Financial Calendar başarıyla yüklendi")
except Exception as e:
    print(f"Financial Calendar yüklenemedi: {e}")
    financial_calendar = None

# Initialize Financial Alert System
try:
    from financial_alerts import FinancialAlertSystem
    financial_alert_system = FinancialAlertSystem()
    print("Financial Alert System başarıyla yüklendi")
except Exception as e:
    print(f"Financial Alert System yüklenemedi: {e}")
    financial_alert_system = None

# Streamlit sayfa konfigürasyonu
st.set_page_config(
    page_title="KCHOL Finansal Asistan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #06b6d4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #06b6d4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Session state başlatma
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = {}

# Model yükleme
@st.cache_resource
def load_model():
    try:
        with open('model/kchol_xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None

# Gemini AI ile genel soruları yanıtlama
def get_gemini_response(user_message, context=""):
    try:
        if any(word in user_message.lower() for word in ['tahmin', 'fiyat', 'ne olacak', 'yükselir mi', 'düşer mi']):
            system_prompt = f"""
Sen profesyonel bir finans analisti olarak KCHOL hisse senedi fiyat tahmini yapıyorsun.

Aşağıdaki verileri kullanarak net, anlaşılır ve profesyonel bir fiyat tahmini yanıtı ver:

{context}

Yanıt kuralları:
1. Sadece Türkçe yanıt ver
2. Emoji kullanma
3. Düzyazı şeklinde yaz
4. ChatGPT tarzında net ve kısa cevaplar ver
5. Teknik jargon kullanma, anlaşılır dil kullan
6. Yatırım tavsiyesi verme, sadece analiz sun
7. Risk uyarısı ekle
8. Maksimum 3-4 paragraf yaz
9. Hata mesajı verme, sadece analiz yap

Kullanıcı sorusu: {user_message}
"""
        else:
            system_prompt = f"""
Sen Türkçe konuşan bir finans ve yatırım asistanısın. KCHOL hisse senedi ve genel finans konularında uzman bilgi veriyorsun.

Kullanıcı sorusu: {user_message}

Lütfen aşağıdaki kurallara uygun olarak yanıt ver:
1. Sadece Türkçe yanıt ver
2. Emoji kullanma
3. Düzyazı şeklinde yaz
4. Finansal tavsiye verme, sadece bilgilendirici ol
5. KCHOL hisse senedi hakkında sorulara özel önem ver
6. Kısa ve öz yanıtlar ver
7. Profesyonel ve anlaşılır dil kullan
8. Hata mesajı verme, sadece bilgi ver

{context}
"""
        
        response = gemini_model.generate_content(system_prompt)
        response_text = response.text.strip()
        
        if "Üzgünüm" in response_text or "şu anda yanıt veremiyorum" in response_text or "error" in response_text.lower():
            return None
            
        return response_text
    except Exception as e:
        print(f"Gemini API hatası: {e}")
        return None

# Hisse verisi alma ve özellik çıkarma
@st.cache_data(ttl=300)  # 5 dakika cache
def get_stock_data(symbol='KCHOL.IS', days=300):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(symbol, start_date, end_date, progress=False)
        
        if df.empty:
            return None
            
        # Sütun isimlerini düzenleme
        df.columns = ['_'.join(col).lower() for col in df.columns]
        df.columns = [col.split('_')[0] for col in df.columns]
        
        # Teknik indikatörler
        df['SMA200'] = TA.SMA(df, 200)
        df['RSI'] = TA.RSI(df)
        df['ATR'] = TA.ATR(df)
        df['BBWidth'] = TA.BBWIDTH(df)
        df['Williams'] = TA.WILLIAMS(df)
        
        # NaN değerleri temizleme
        df = df.dropna()
        
        if len(df) < 1:
            return None
            
        return df
    except Exception as e:
        print(f"Veri alma hatası: {e}")
        return None

# Grafik oluşturma fonksiyonu
def create_price_chart(df, symbol, current_price, predicted_price):
    """Hisse senedi fiyat grafiği oluştur"""
    try:
        # Son 30 günlük veri
        recent_data = df.tail(30)
        
        # Modern fintech renk paleti
        colors = {
            'primary': '#2563eb',      # Mavi (ana renk)
            'secondary': '#7c3aed',    # Mor (ikincil)
            'success': '#059669',      # Yeşil (başarı)
            'warning': '#dc2626',      # Kırmızı (uyarı)
            'info': '#0891b2',         # Turkuaz (bilgi)
            'neutral': '#6b7280',      # Gri (nötr)
            'accent': '#f59e0b',       # Turuncu (vurgu)
            'background': '#f8fafc'    # Açık gri (arka plan)
        }
        
        fig = go.Figure()
        
        # Fiyat çizgisi - Gradient renk efekti
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['close'],
            mode='lines',
            name='Fiyat',
            line=dict(
                color=colors['primary'], 
                width=3,
                shape='spline'  # Yumuşak çizgi
            ),
            fill='tonexty',
            fillcolor=f'rgba(37, 99, 235, 0.1)'  # Hafif mavi dolgu
        ))
        
        # Mevcut fiyat çizgisi
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color=colors['success'],
            line_width=2,
            annotation_text=f"Mevcut Fiyat: {current_price} TL",
            annotation_position="top right"
        )
        
        # Tahmin fiyatı çizgisi
        fig.add_hline(
            y=predicted_price,
            line_dash="dash",
            line_color=colors['warning'],
            line_width=2,
            annotation_text=f"Tahmin: {predicted_price:.2f} TL",
            annotation_position="top left"
        )
        
        # 200 günlük ortalama
        if 'SMA200' in recent_data.columns:
            sma200 = recent_data['SMA200'].dropna()
            if not sma200.empty:
                fig.add_trace(go.Scatter(
                    x=sma200.index,
                    y=sma200,
                    mode='lines',
                    name='SMA200',
                    line=dict(
                        color=colors['accent'], 
                        width=2, 
                        dash='dot'
                    )
                ))
        
        fig.update_layout(
            title=dict(
                text=f'📈 {symbol} Fiyat Grafiği ve Tahmin',
                font=dict(size=18, color=colors['primary'])
            ),
            xaxis_title='Tarih',
            yaxis_title='Fiyat (TL)',
            template='plotly_white',
            height=450,
            showlegend=True,
            plot_bgcolor=colors['background'],
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        print(f"Grafik oluşturma hatası: {e}")
        return None

# Teknik analiz grafiği oluşturma fonksiyonu
def create_technical_chart(df, symbol):
    """Teknik analiz grafiği oluştur"""
    try:
        # Son 50 günlük veri
        recent_data = df.tail(50)
        
        # Modern fintech renk paleti
        colors = {
            'primary': '#2563eb',      # Mavi (ana renk)
            'secondary': '#7c3aed',    # Mor (ikincil)
            'success': '#059669',      # Yeşil (başarı)
            'warning': '#dc2626',      # Kırmızı (uyarı)
            'info': '#0891b2',         # Turkuaz (bilgi)
            'neutral': '#6b7280',      # Gri (nötr)
            'accent': '#f59e0b',       # Turuncu (vurgu)
            'background': '#f8fafc'    # Açık gri (arka plan)
        }
        
        # Alt grafikler oluştur
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'📊 {symbol} Fiyat ve Bollinger Bantları', 
                '📈 RSI Göstergesi', 
                '📉 MACD Göstergesi'
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Fiyat ve Bollinger Bantları
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['close'],
            mode='lines',
            name='Fiyat',
            line=dict(
                color=colors['primary'], 
                width=3,
                shape='spline'
            ),
            fill='tonexty',
            fillcolor=f'rgba(37, 99, 235, 0.1)'
        ), row=1, col=1)
        
        # Bollinger Bantları (eğer varsa)
        if 'BB_upper' in recent_data.columns and 'BB_lower' in recent_data.columns:
            bb_upper = recent_data['BB_upper'].dropna()
            bb_lower = recent_data['BB_lower'].dropna()
            bb_middle = recent_data['BB_middle'].dropna()
            
            if not bb_upper.empty:
                fig.add_trace(go.Scatter(
                    x=bb_upper.index,
                    y=bb_upper,
                    mode='lines',
                    name='BB Üst',
                    line=dict(
                        color=colors['warning'], 
                        width=2, 
                        dash='dash'
                    )
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=bb_lower.index,
                    y=bb_lower,
                    mode='lines',
                    name='BB Alt',
                    line=dict(
                        color=colors['warning'], 
                        width=2, 
                        dash='dash'
                    ),
                    fill='tonexty',
                    fillcolor=f'rgba(220, 38, 38, 0.1)'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=bb_middle.index,
                    y=bb_middle,
                    mode='lines',
                    name='BB Orta',
                    line=dict(
                        color=colors['accent'], 
                        width=2
                    )
                ), row=1, col=1)
        
        # RSI
        if 'RSI' in recent_data.columns:
            rsi = recent_data['RSI'].dropna()
            if not rsi.empty:
                fig.add_trace(go.Scatter(
                    x=rsi.index,
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(
                        color=colors['secondary'], 
                        width=3,
                        shape='spline'
                    ),
                    fill='tonexty',
                    fillcolor=f'rgba(124, 58, 237, 0.1)'
                ), row=2, col=1)
                
                # RSI seviyeleri
                fig.add_hline(
                    y=70, 
                    line_dash="dash", 
                    line_color=colors['warning'], 
                    line_width=2,
                    row=2, col=1,
                    annotation_text="Aşırı Alım (70)"
                )
                fig.add_hline(
                    y=30, 
                    line_dash="dash", 
                    line_color=colors['success'], 
                    line_width=2,
                    row=2, col=1,
                    annotation_text="Aşırı Satım (30)"
                )
                fig.add_hline(
                    y=50, 
                    line_dash="dot", 
                    line_color=colors['neutral'], 
                    line_width=1,
                    row=2, col=1
                )
        
        # MACD (eğer varsa)
        if 'MACD' in recent_data.columns and 'MACD_signal' in recent_data.columns:
            macd = recent_data['MACD'].dropna()
            macd_signal = recent_data['MACD_signal'].dropna()
            
            if not macd.empty:
                fig.add_trace(go.Scatter(
                    x=macd.index,
                    y=macd,
                    mode='lines',
                    name='MACD',
                    line=dict(
                        color=colors['info'], 
                        width=3,
                        shape='spline'
                    )
                ), row=3, col=1)
                
                if not macd_signal.empty:
                    fig.add_trace(go.Scatter(
                        x=macd_signal.index,
                        y=macd_signal,
                        mode='lines',
                        name='MACD Signal',
                        line=dict(
                            color=colors['warning'], 
                            width=2
                        )
                    ), row=3, col=1)
        
        fig.update_layout(
            title=dict(
                text=f'📊 {symbol} Teknik Analiz Grafiği',
                font=dict(size=20, color=colors['primary'])
            ),
            height=800,
            showlegend=True,
            template='plotly_white',
            plot_bgcolor=colors['background'],
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Y ekseni etiketleri
        fig.update_yaxes(
            title_text="Fiyat (TL)", 
            row=1, col=1,
            title_font=dict(color=colors['primary'], size=12)
        )
        fig.update_yaxes(
            title_text="RSI", 
            row=2, col=1,
            title_font=dict(color=colors['secondary'], size=12)
        )
        fig.update_yaxes(
            title_text="MACD", 
            row=3, col=1,
            title_font=dict(color=colors['info'], size=12)
        )
        
        return fig
    except Exception as e:
        print(f"Teknik analiz grafiği oluşturma hatası: {e}")
        return None

# Tahmin fonksiyonu
def create_model_explanation(X, features, predicted_price, current_price):
    """Model tahminini açıklayan basit analiz"""
    try:
        feature_values = X[0] if len(X.shape) > 1 else X
        
        explanations = []
        
        # Fiyat verileri analizi
        close_price = feature_values[features.index('close')]
        high_price = feature_values[features.index('high')]
        low_price = feature_values[features.index('low')]
        open_price = feature_values[features.index('open')]
        volume = feature_values[features.index('volume')]
        
        # Teknik göstergeler
        sma200 = feature_values[features.index('SMA200')]
        rsi = feature_values[features.index('RSI')]
        atr = feature_values[features.index('ATR')]
        bbwidth = feature_values[features.index('BBWidth')]
        williams = feature_values[features.index('Williams')]
        
        # Fiyat pozisyonu analizi
        if close_price > sma200:
            explanations.append(f"Kapanış fiyatı ({close_price:.2f} TL) 200 günlük ortalamanın ({sma200:.2f} TL) üzerinde - Yükseliş trendi")
        else:
            explanations.append(f"Kapanış fiyatı ({close_price:.2f} TL) 200 günlük ortalamanın ({sma200:.2f} TL) altında - Düşüş trendi")
        
        # RSI analizi
        if rsi > 70:
            explanations.append(f"RSI ({rsi:.1f}) aşırı alım bölgesinde - Düşüş riski")
        elif rsi < 30:
            explanations.append(f"RSI ({rsi:.1f}) aşırı satım bölgesinde - Yükseliş fırsatı")
        else:
            explanations.append(f"RSI ({rsi:.1f}) nötr bölgede - Trend devam edebilir")
        
        # Volatilite analizi
        if atr > 5:
            explanations.append(f"Yüksek volatilite (ATR: {atr:.2f}) - Fiyat hareketleri büyük olabilir")
        else:
            explanations.append(f"Düşük volatilite (ATR: {atr:.2f}) - Fiyat hareketleri sınırlı olabilir")
        
        # Bollinger Bant analizi
        if bbwidth > 0.2:
            explanations.append(f"Geniş Bollinger Bantları ({bbwidth:.3f}) - Volatilite artıyor")
        else:
            explanations.append(f"Dar Bollinger Bantları ({bbwidth:.3f}) - Volatilite azalıyor")
        
        # Williams %R analizi
        if williams < -80:
            explanations.append(f"Williams %R ({williams:.1f}) aşırı satım - Yükseliş sinyali")
        elif williams > -20:
            explanations.append(f"Williams %R ({williams:.1f}) aşırı alım - Düşüş sinyali")
        else:
            explanations.append(f"Williams %R ({williams:.1f}) nötr bölge")
        
        # Hacim analizi
        avg_volume = volume / 1000000  # Milyon cinsinden
        if avg_volume > 10:
            explanations.append(f"Yüksek işlem hacmi ({avg_volume:.1f}M) - Güçlü trend")
        else:
            explanations.append(f"Düşük işlem hacmi ({avg_volume:.1f}M) - Zayıf trend")
        
        # Tahmin yönü analizi
        if predicted_price > current_price:
            trend_direction = "YÜKSELİŞ"
            confidence = "Yüksek" if abs(predicted_price - current_price) > 5 else "Orta"
        else:
            trend_direction = "DÜŞÜŞ"
            confidence = "Yüksek" if abs(predicted_price - current_price) > 5 else "Orta"
        
        return {
            'trend_direction': trend_direction,
            'confidence': confidence,
            'explanations': explanations,
            'key_factors': {
                'price_vs_sma200': "Yukarı" if close_price > sma200 else "Aşağı",
                'rsi_signal': "Aşırı alım" if rsi > 70 else "Aşırı satım" if rsi < 30 else "Nötr",
                'volatility': "Yüksek" if atr > 5 else "Düşük",
                'volume_strength': "Güçlü" if avg_volume > 10 else "Zayıf"
            }
        }
        
    except Exception as e:
        print(f"Model açıklama hatası: {e}")
        return {
            'trend_direction': "Belirsiz",
            'confidence': "Düşük",
            'explanations': ["Model açıklaması oluşturulamadı"],
            'key_factors': {}
        }

def predict_price(model, df):
    try:
        if df is None:
            return None, "Veri bulunamadı"
            
        if len(df) < 1:
            return None, f"Yeterli veri bulunamadı. Mevcut veri: {len(df)} satır"
        
        # Son veriyi al
        latest_data = df.iloc[-1:].copy()
        
        # Gerekli özellikler
        features = ['close', 'high', 'low', 'open', 'volume', 'SMA200', 'RSI', 'ATR', 'BBWidth', 'Williams']
        
        # Eksik özellikleri kontrol et
        missing_features = [f for f in features if f not in latest_data.columns]
        if missing_features:
            return None, f"Eksik özellikler: {missing_features}"
        
        # Tahmin için veriyi hazırla
        X = latest_data[features].values
        
        # Tahmin yap
        prediction = model.predict(X)[0]
        
        current_price = latest_data['close'].iloc[0]
        change = prediction - current_price
        change_percent = (change / current_price) * 100
        
        # Tahmin tarihini hesapla (hafta sonu kontrolü ile)
        tomorrow = datetime.now() + timedelta(days=1)
        if tomorrow.weekday() >= 5:  # Cumartesi veya Pazar
            while tomorrow.weekday() >= 5:
                tomorrow = tomorrow + timedelta(days=1)
        
        # Model açıklaması oluştur
        model_explanation = create_model_explanation(X, features, prediction, current_price)
        
        result = {
            'current_price': float(round(current_price, 2)),
            'predicted_price': float(round(prediction, 2)),
            'change': float(round(change, 2)),
            'change_percent': float(round(change_percent, 2)),
            'prediction_date': tomorrow.strftime('%Y-%m-%d'),
            'model_explanation': model_explanation
        }
        
        return result, None
        
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return None, f"Tahmin hatası: {e}"

# Ana sayfa - Chatbot
def main_page():
    st.markdown('<h1 class="main-header">🤖 BIST Finansal Asistan</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Tüm BIST hisse senetleri için akıllı analiz ve yatırım tavsiyeleri</p>', unsafe_allow_html=True)
    
    # Sidebar - Hızlı Erişim
    with st.sidebar:
        st.markdown("### Popüler Hisse Senetleri")
        
        popular_stocks = [
            ("KCHOL", "Koç Holding"),
            ("THYAO", "Türk Hava Yolları"),
            ("GARAN", "Garanti BBVA"),
            ("AKBNK", "Akbank"),
            ("ASELS", "Aselsan"),
            ("EREGL", "Ereğli Demir Çelik")
        ]
        
        for symbol, name in popular_stocks:
            if st.button(f"{symbol} - {name}", use_container_width=True, key=f"popular_{symbol}"):
                # Doğrudan mesajı işle
                st.session_state.chat_history.append({
                    'sender': 'user',
                    'message': f"{symbol} fiyat tahmini yap",
                    'timestamp': datetime.now()
                })
                
                with st.spinner("Düşünüyorum..."):
                    bot_response = process_message(f"{symbol} fiyat tahmini yap")
                
                st.session_state.chat_history.append({
                    'sender': 'bot',
                    'message': bot_response,
                    'timestamp': datetime.now()
                })
                st.rerun()
        
        
        if st.button("📈 Fiyat Tahmini", use_container_width=True, key="quick_price_prediction"):
            # Doğrudan mesajı işle
            st.session_state.chat_history.append({
                'sender': 'user',
                'message': "KCHOL fiyat tahmini yap",
                'timestamp': datetime.now()
            })
            
            with st.spinner("Düşünüyorum..."):
                bot_response = process_message("KCHOL fiyat tahmini yap")
            
            st.session_state.chat_history.append({
                'sender': 'bot',
                'message': bot_response,
                'timestamp': datetime.now()
            })
            st.rerun()
        
        if st.button("📊 Teknik Analiz", use_container_width=True, key="quick_tech_analysis"):
            # Doğrudan mesajı işle
            st.session_state.chat_history.append({
                'sender': 'user',
                'message': "KCHOL teknik analiz yap",
                'timestamp': datetime.now()
            })
            
            with st.spinner("Düşünüyorum..."):
                bot_response = process_message("KCHOL teknik analiz yap")
            
            st.session_state.chat_history.append({
                'sender': 'bot',
                'message': bot_response,
                'timestamp': datetime.now()
            })
            st.rerun()
        
        if st.button("📰 Haber Analizi", use_container_width=True, key="quick_news_analysis"):
            # Doğrudan mesajı işle
            st.session_state.chat_history.append({
                'sender': 'user',
                'message': "KCHOL haber analizi yap",
                'timestamp': datetime.now()
            })
            
            with st.spinner("Düşünüyorum..."):
                bot_response = process_message("KCHOL haber analizi yap")
            
            st.session_state.chat_history.append({
                'sender': 'bot',
                'message': bot_response,
                'timestamp': datetime.now()
            })
            st.rerun()
        
        if st.button("💼 Portföy Simülasyonu", use_container_width=True, key="quick_portfolio_sim"):
            # Doğrudan mesajı işle
            st.session_state.chat_history.append({
                'sender': 'user',
                'message': "KCHOL'a 6 ay önce 10.000 TL yatırsaydım ne olurdu?",
                'timestamp': datetime.now()
            })
            
            with st.spinner("Düşünüyorum..."):
                bot_response = process_message("KCHOL'a 6 ay önce 10.000 TL yatırsaydım ne olurdu?")
            
            st.session_state.chat_history.append({
                'sender': 'bot',
                'message': bot_response,
                'timestamp': datetime.now()
            })
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📋 Menü")
        
        if st.button("🏠 Ana Sayfa", use_container_width=True, key="menu_home"):
            st.session_state.page = "Ana Sayfa"
            st.rerun()
        
        if st.button("💼 Portföy Yönetimi", use_container_width=True, key="menu_portfolio"):
            st.session_state.page = "Portföy Yönetimi"
            st.rerun()
        
        if st.button("📅 Finansal Takvim", use_container_width=True, key="menu_calendar"):
            st.session_state.page = "Finansal Takvim"
            st.rerun()
        
        if st.button("📊 Teknik Analiz", use_container_width=True, key="menu_technical"):
            st.session_state.page = "Teknik Analiz"
            st.rerun()
        
        if st.button("🔔 Alarm Yönetimi", use_container_width=True, key="menu_alerts"):
            st.session_state.page = "Alarm Yönetimi"
            st.rerun()
    
    # Hisse seçici ve hızlı erişim
    st.markdown("### 🎯 Hızlı Analiz")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stock = st.selectbox(
            "Hisse Seçin",
            ["KCHOL", "THYAO", "GARAN", "AKBNK", "ASELS", "EREGL", "SASA", "ISCTR", "BIMAS", "ALARK", "TUPRS", "PGSU", "KRMD", "TAVHL", "DOAS", "TOASO", "FROTO", "VESTL", "YAPI", "QNBFB", "HALKB", "VAKBN", "SISE", "KERVN"],
            key="stock_selector"
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Analiz Türü",
            ["Fiyat Tahmini", "Teknik Analiz", "Haber Analizi", "Simülasyon"],
            key="analysis_type"
        )
    
    with col3:
        if st.button("Analiz Yap", type="primary", key="quick_analysis"):
            if analysis_type == "Fiyat Tahmini":
                message = f"{selected_stock} fiyat tahmini yap"
            elif analysis_type == "Teknik Analiz":
                message = f"{selected_stock} teknik analiz yap"
            elif analysis_type == "Haber Analizi":
                message = f"{selected_stock} haber analizi yap"
            elif analysis_type == "Simülasyon":
                message = f"{selected_stock}'a 6 ay önce 10.000 TL yatırsaydım ne olurdu?"
            
            # Doğrudan mesajı işle
            st.session_state.chat_history.append({
                'sender': 'user',
                'message': message,
                'timestamp': datetime.now()
            })
            
            with st.spinner("Düşünüyorum..."):
                bot_response = process_message(message)
            
            st.session_state.chat_history.append({
                'sender': 'bot',
                'message': bot_response,
                'timestamp': datetime.now()
            })
            st.rerun()
    
    # Örnek sorular
    st.markdown("### 💡 Örnek Sorular")
    
    example_questions = [
        "THYAO fiyat tahmini yap",
        "GARAN teknik analiz yap", 
        "AKBNK'a 6 ay önce 10.000 TL yatırsaydım ne olurdu?",
        "ASELS RSI değeri nedir?",
        "EREGL bilançosu ne zaman?",
        "Tüm hisselerin RSI'si 70 üstü olanları listele"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(question, key=f"example_{i}", use_container_width=True):
                # Doğrudan mesajı işle
                st.session_state.chat_history.append({
                    'sender': 'user',
                    'message': question,
                    'timestamp': datetime.now()
                })
                
                with st.spinner("Düşünüyorum..."):
                    bot_response = process_message(question)
                
                st.session_state.chat_history.append({
                    'sender': 'bot',
                    'message': bot_response,
                    'timestamp': datetime.now()
                })
                st.rerun()
    
    # Chat arayüzü
    st.markdown("### 💬 Sohbet")
    
    # Chat geçmişi
    for message in st.session_state.chat_history:
        if message['sender'] == 'user':
            st.markdown(f'<div class="chat-message user-message"><strong>Siz:</strong> {message["message"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><strong>Asistan:</strong> {message["message"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    chat_input = st.text_input(
        "Mesajınızı yazın...",
        key="chat_input",
        placeholder="Örn: KCHOL fiyat tahmini yap, teknik analiz göster, portföy simülasyonu...",
        value=""
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        send_button = st.button("Gönder", type="primary", key="send_message")
    
    with col2:
        clear_button = st.button("Temizle", key="clear_chat")
    
    # Mesaj gönderme
    if send_button and chat_input and len(chat_input.strip()) > 0:
        # Kullanıcı mesajını geçmişe ekle
        st.session_state.chat_history.append({
            'sender': 'user',
            'message': chat_input,
            'timestamp': datetime.now()
        })
        
        # Bot yanıtını oluştur
        with st.spinner("Düşünüyorum..."):
            bot_response = process_message(chat_input)
        
        # Bot yanıtını geçmişe ekle
        st.session_state.chat_history.append({
            'sender': 'bot',
            'message': bot_response,
            'timestamp': datetime.now()
        })
        
        # Sayfayı yenile (input otomatik temizlenecek)
        st.rerun()
    
    # Chat temizleme
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

def process_message(message):
    """Mesajı işle ve yanıt döndür"""
    message_lower = message.lower()
    
    # Model yükleme
    model = load_model()
    if model is None:
        return 'Üzgünüm, model şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.'
    
    # Fiyat tahmini
    if any(word in message_lower for word in ['tahmin', 'fiyat', 'ne olacak', 'yükselir mi', 'düşer mi']):
        # Hisse kodunu mesajdan çıkar
        hisse_kodu = 'KCHOL'  # Varsayılan
        for symbol in ['KCHOL', 'THYAO', 'GARAN', 'AKBNK', 'ASELS', 'EREGL', 'SASA', 'ISCTR', 'BIMAS', 'ALARK', 'TUPRS', 'PGSU', 'KRMD', 'TAVHL', 'DOAS', 'TOASO', 'FROTO', 'VESTL', 'YAPI', 'QNBFB', 'HALKB', 'VAKBN', 'SISE', 'KERVN']:
            if symbol.lower() in message_lower:
                hisse_kodu = symbol
                break
        
        # Hisse verisini al
        symbol_with_suffix = f"{hisse_kodu}.IS"
        df = get_stock_data(symbol_with_suffix)
        if df is None:
            return f'{hisse_kodu} hisse verisi alınamadı. Lütfen daha sonra tekrar deneyin.'
        
        result, error = predict_price(model, df)
        if error:
            return f'Tahmin yapılamadı: {error}'
        
        trend_text = "Yükseliş bekleniyor!" if result['change'] > 0 else "Düşüş bekleniyor!" if result['change'] < 0 else "Fiyat sabit kalabilir"
        
        # Grafik oluştur
        fig = create_price_chart(df, hisse_kodu, result['current_price'], result['predicted_price'])
        st.plotly_chart(fig, use_container_width=True)
        
        response = f"""**{hisse_kodu} Hisse Senedi Fiyat Tahmini**

{hisse_kodu} hisse senedi şu anda **{result['current_price']} TL** seviyesinde işlem görüyor. 

Teknik analiz sonuçlarına göre, hisse senedinin **{result['predicted_price']:.2f} TL** seviyesine **{result['change']:+.2f} TL** ({result['change_percent']:+.2f}%) değişimle ulaşması bekleniyor. {trend_text}

**Tahmin Tarihi:** {result['prediction_date']}

⚠️ **RİSK UYARISI:** Bu analiz sadece teknik göstergelere dayalıdır ve yatırım tavsiyesi değildir. Hisse senedi yatırımları risklidir ve kayıplara yol açabilir."""
        
        return response
    
    # Teknik analiz
    elif any(word in message_lower for word in ['teknik analiz', 'teknik', 'grafik', 'indikatör', 'rsi', 'macd']):
        if technical_analysis_engine:
            try:
                # Hisse kodunu mesajdan çıkar
                hisse_kodu = 'KCHOL'  # Varsayılan
                for symbol in ['KCHOL', 'THYAO', 'GARAN', 'AKBNK', 'ASELS', 'EREGL', 'SASA', 'ISCTR', 'BIMAS', 'ALARK', 'TUPRS', 'PGSU', 'KRMD', 'TAVHL', 'DOAS', 'TOASO', 'FROTO', 'VESTL', 'YAPI', 'QNBFB', 'HALKB', 'VAKBN', 'SISE', 'KERVN']:
                    if symbol.lower() in message_lower:
                        hisse_kodu = symbol
                        break
                
                result = technical_analysis_engine.process_technical_analysis_request(message)
                if result.get('error'):
                    return f'Teknik analiz hatası: {result["error"]}'
                
                # Teknik analiz grafiği oluştur
                symbol_with_suffix = f"{hisse_kodu}.IS"
                df = get_stock_data(symbol_with_suffix)
                if df is not None:
                    tech_fig = create_technical_chart(df, hisse_kodu)
                    if tech_fig:
                        st.plotly_chart(tech_fig, use_container_width=True)
                
                # Teknik analiz modülünden gelen grafikleri göster
                if 'charts' in result:
                    for chart in result['charts']:
                        if chart.get('type') == 'line' and 'data' in chart:
                            # HTML img tagını temizle ve sadece base64 veriyi al
                            img_data = chart['data']
                            if img_data.startswith('<img src="data:image/png;base64,'):
                                # Base64 veriyi çıkar
                                start = img_data.find('base64,') + 7
                                end = img_data.find('"', start)
                                base64_data = img_data[start:end]
                                
                                # Streamlit'te göster
                                st.image(f"data:image/png;base64,{base64_data}", 
                                        caption=chart.get('title', ''), 
                                        use_column_width=True)
                            else:
                                # HTML olarak göster
                                st.markdown(img_data, unsafe_allow_html=True)
                
                # Eğer analysis içinde HTML img tagları varsa onları da işle
                analysis_text = result.get('analysis', '')
                if '<img src="data:image/png;base64,' in analysis_text:
                    # HTML img taglarını bul ve işle
                    import re
                    img_pattern = r'<img src="data:image/png;base64,([^"]+)"[^>]*>'
                    matches = re.findall(img_pattern, analysis_text)
                    
                    for base64_data in matches:
                        st.image(f"data:image/png;base64,{base64_data}", 
                                use_column_width=True)
                    
                    # HTML img taglarını temizle
                    analysis_text = re.sub(img_pattern, '', analysis_text)
                    result['analysis'] = analysis_text
                
                response = f"""**{hisse_kodu} Teknik Analiz Raporu**

{result.get('analysis', '')}

{result.get('summary', '')}

⚠️ **RİSK UYARISI:** Bu analiz sadece teknik göstergelere dayalıdır ve yatırım tavsiyesi değildir."""
                
                return response
            except Exception as e:
                return f'Teknik analiz yapılamadı: {str(e)}'
        else:
            return 'Teknik analiz motoru şu anda kullanılamıyor.'
    
    # Hisse simülasyonu
    elif any(word in message_lower for word in ['simülasyon', 'simulasyon', 'simulation', 'ne olurdu', 'olurdu', 'kaç para']):
        if hisse_simulasyon:
            try:
                # Hisse kodunu mesajdan çıkar
                hisse_kodu = 'KCHOL'  # Varsayılan
                for symbol in ['KCHOL', 'THYAO', 'GARAN', 'AKBNK', 'ASELS', 'EREGL', 'SASA', 'ISCTR', 'BIMAS', 'ALARK', 'TUPRS', 'PGSU', 'KRMD', 'TAVHL', 'DOAS', 'TOASO', 'FROTO', 'VESTL', 'YAPI', 'QNBFB', 'HALKB', 'VAKBN', 'SISE', 'KERVN']:
                    if symbol.lower() in message_lower:
                        hisse_kodu = symbol
                        break
                
                # Simülasyon için hisse kodunu .IS ile birleştir
                hisse_kodu_with_suffix = f"{hisse_kodu}.IS"
                tarih = "6 ay önce"
                tutar = 10000.0
                
                sim_result = hisse_simulasyon(hisse_kodu_with_suffix, tarih, tutar)
                
                if 'hata' not in sim_result:
                    response = f"""**📊 {hisse_kodu} Hisse Senedi Simülasyon Sonucu**

**Simülasyon Detayları:**
• **Hisse:** {hisse_kodu}
• **Başlangıç Tarihi:** {sim_result['başlangıç tarihi']}
• **Yatırım Tutarı:** {tutar:,.2f} TL

**Fiyat Analizi:**
• **Başlangıç Fiyatı:** {sim_result['başlangıç fiyatı']} TL
• **Güncel Fiyat:** {sim_result['güncel fiyat']} TL
• **Alınan Lot:** {sim_result['alınan lot']} adet

**Sonuç:**
• **Şu Anki Değer:** {sim_result['şu anki değer']:,.2f} TL
• **Net Kazanç:** {sim_result['net kazanç']:,.2f} TL
• **Getiri Oranı:** %{sim_result['getiri %']:.2f}

{'🟢 **KARLILIK**' if sim_result['net kazanç'] > 0 else '🔴 **ZARAR**' if sim_result['net kazanç'] < 0 else '⚪ **BREAKEVEN**'}"""
                else:
                    response = f"❌ Simülasyon hatası: {sim_result['hata']}"
                
                return response
            except Exception as e:
                return f'Hisse simülasyonu yapılamadı: {str(e)}'
        else:
            return 'Hisse simülasyon sistemi şu anda kullanılamıyor.'
    
    # Yardım
    elif any(word in message_lower for word in ['yardım', 'help', 'nasıl', 'ne yapabilir']):
        return """**BIST Finansal Asistanı**

Size şu konularda yardımcı olabilirim:

📊 **Teknik Analiz:** "Teknik analiz yap", "RSI göster", "MACD analizi"
📈 **Fiyat Tahmini:** "Fiyat tahmini yap", "Ne olacak", "Yükselir mi"
📰 **Haber Analizi:** "Haber analizi yap", "Son haberler"
📅 **Finansal Takvim:** "Bilanço ne zaman", "Temettü tarihi", "Genel kurul"
💡 **Öneriler:** Yatırım kararlarınız için veri tabanlı öneriler
🔍 **Finansal Q&A:** Doğal dil ile finansal sorular
🎯 **Hisse Simülasyonu:** Geçmiş yatırım senaryoları

**Desteklenen Hisse Senetleri:**
KCHOL, THYAO, GARAN, AKBNK, ASELS, EREGL, SASA, ISCTR, BIMAS, ALARK, TUPRS, PGSU, KRMD, TAVHL, DOAS, TOASO, FROTO, VESTL, YAPI, QNBFB, HALKB, VAKBN, SISE, KERVN

**Örnek Sorular:**
• "THYAO fiyat tahmini yap"
• "GARAN teknik analiz göster"
• "EREGL bilançosu ne zaman?"
• "AKBNK'a 6 ay önce 10.000 TL yatırsaydım ne olurdu?"
• "ASELS RSI değeri nedir?"

Sadece sorunuzu yazın, size yardımcı olayım!"""
    
    # Selamlaşma
    elif any(word in message_lower for word in ['merhaba', 'selam', 'hi', 'hello']) and len(message.split()) <= 3:
        return 'Merhaba! Ben KCHOL hisse senedi fiyat tahmin asistanınız. Size yardımcı olmak için buradayım. Fiyat tahmini yapmak ister misiniz?'
    
    # Finansal takvim soruları
    elif any(word in message_lower for word in ['bilanço', 'temettü', 'genel kurul', 'faaliyet raporu', 'ne zaman', 'tarih', 'takvim']):
        if financial_calendar:
            try:
                # Hisse kodunu mesajdan çıkar
                hisse_kodu = None
                for symbol in ['KCHOL', 'THYAO', 'GARAN', 'AKBNK', 'ASELS', 'EREGL', 'SASA', 'ISCTR', 'BIMAS', 'ALARK', 'TUPRS', 'PGSU', 'KRMD', 'TAVHL', 'DOAS', 'TOASO', 'FROTO', 'VESTL', 'YAPI', 'QNBFB', 'HALKB', 'VAKBN', 'SISE', 'KERVN']:
                    if symbol.lower() in message_lower:
                        hisse_kodu = symbol
                        break
                
                if not hisse_kodu:
                    hisse_kodu = 'KCHOL'  # Varsayılan
                
                company_data = financial_calendar.get_company_events(hisse_kodu)
                
                if company_data and 'events' in company_data and len(company_data['events']) > 0:
                    events = company_data['events']
                    response = f"**{hisse_kodu} Finansal Takvim**\n\n"
                    
                    for event in events:
                        event_date = datetime.strptime(event['date'], '%Y-%m-%d').strftime('%d.%m.%Y')
                        status_emoji = "🟢" if event['status'] == 'tamamlandı' else "🟡" if event['status'] == 'bekliyor' else "🔴"
                        
                        response += f"{status_emoji} **{event['type'].title()}**\n"
                        response += f"📅 Tarih: {event_date}\n"
                        response += f"📝 Açıklama: {event['description']}\n"
                        response += f"📊 Durum: {event['status'].title()}\n\n"
                    
                    response += "💡 **Not:** Tarihler yaklaşık olup, şirket duyurularına göre değişebilir."
                    return response
                else:
                    return f"{hisse_kodu} için finansal takvim bilgisi bulunamadı. Lütfen daha sonra tekrar deneyin."
            except Exception as e:
                return f'Finansal takvim bilgisi alınamadı: {str(e)}'
        else:
            return 'Finansal takvim sistemi şu anda kullanılamıyor.'
    
    # Genel sorular
    else:
        try:
            if gemini_model:
                gemini_response = get_gemini_response(message)
                if gemini_response:
                    return gemini_response
            
            # Fallback to Document RAG Agent
            if document_rag_agent:
                return document_rag_agent.process_query(message)
            else:
                return 'Üzgünüm, şu anda size yardımcı olamıyorum. Lütfen daha sonra tekrar deneyin.'
        except Exception as e:
            return f'Bir hata oluştu: {str(e)}'

# Portföy Yönetimi Sayfası
def portfolio_page():
    st.markdown('<h1 class="main-header">💼 Portföy Yönetimi</h1>', unsafe_allow_html=True)
    
    if not portfolio_manager:
        st.error("Portföy yöneticisi kullanılamıyor.")
        return
    
    # Portföy özeti
    user_id = st.session_state.current_session_id
    portfolio_summary = portfolio_manager.get_portfolio_summary(user_id)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Toplam Değer", f"{portfolio_summary.get('total_value', 0):,.2f} TL")
    
    with col2:
        st.metric("Toplam Kazanç", f"{portfolio_summary.get('total_gain', 0):,.2f} TL")
    
    with col3:
        st.metric("Getiri Oranı", f"{portfolio_summary.get('return_percentage', 0):.2f}%")
    
    with col4:
        st.metric("Hisse Sayısı", portfolio_summary.get('stock_count', 0))
    
    # Hisse ekleme formu
    st.markdown("### ➕ Hisse Ekle")
    
    with st.form("add_stock_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("Hisse Kodu", placeholder="KCHOL")
        
        with col2:
            quantity = st.number_input("Miktar", min_value=0.0, step=1.0)
        
        with col3:
            avg_price = st.number_input("Ortalama Fiyat", min_value=0.0, step=0.01)
        
        if st.form_submit_button("Hisse Ekle", type="primary", key="add_stock_submit"):
            if symbol and quantity > 0 and avg_price > 0:
                result = portfolio_manager.add_stock(user_id, symbol.upper(), quantity, avg_price)
                if result['success']:
                    st.success(result['message'])
                    st.rerun()
                else:
                    st.error(result['message'])
            else:
                st.error("Lütfen tüm alanları doldurun.")
    
    # Portföy listesi
    st.markdown("### 📊 Portföy Detayları")
    
    if portfolio_summary.get('stocks'):
        df = pd.DataFrame(portfolio_summary['stocks'])
        st.dataframe(df, use_container_width=True)
        
        # Hisse çıkarma
        st.markdown("### ➖ Hisse Çıkar")
        
        with st.form("remove_stock_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                remove_symbol = st.selectbox("Hisse Kodu", [stock['symbol'] for stock in portfolio_summary['stocks']])
            
            with col2:
                remove_quantity = st.number_input("Çıkarılacak Miktar", min_value=0.0, step=1.0, help="0 girerseniz tüm hisse çıkarılır")
            
            if st.form_submit_button("Hisse Çıkar", type="secondary", key="remove_stock_submit"):
                result = portfolio_manager.remove_stock(user_id, remove_symbol, remove_quantity if remove_quantity > 0 else None)
                if result['success']:
                    st.success(result['message'])
                    st.rerun()
                else:
                    st.error(result['message'])
    else:
        st.info("Portföyünüzde henüz hisse bulunmuyor.")

# Finansal Takvim Sayfası
def calendar_page():
    st.markdown('<h1 class="main-header">📅 Finansal Takvim</h1>', unsafe_allow_html=True)
    
    if not financial_calendar:
        st.error("Finansal takvim kullanılamıyor.")
        return
    
    # Şirket seçimi
    companies = financial_calendar.get_companies()
    selected_company = st.selectbox("Şirket Seçin", companies)
    
    if selected_company:
        company_events = financial_calendar.get_company_events(selected_company)
        
        if company_events and company_events['events']:
            st.markdown(f"### {company_events['company_name']} ({selected_company}) Finansal Takvimi")
            
            for event in company_events['events']:
                with st.expander(f"{event['type'].title()} - {event['date']}"):
                    st.write(f"**Açıklama:** {event['description']}")
                    st.write(f"**Kaynak:** {event['source']}")
                    st.write(f"**Durum:** {event['status']}")
        else:
            st.info(f"{selected_company} için finansal takvim bilgisi bulunamadı.")
    
    # Yaklaşan olaylar
    st.markdown("### 🔔 Yaklaşan Olaylar")
    
    days = st.slider("Kaç gün içindeki olayları göster", 1, 90, 30)
    
    if st.button("Yaklaşan Olayları Getir", key="get_upcoming_events"):
        upcoming_events = financial_calendar.get_upcoming_events(days)
        
        if upcoming_events:
            for event in upcoming_events:
                st.write(f"**{event['company']}** - {event['type']} - {event['date']}")
                st.write(f"{event['description']}")
                st.write("---")
        else:
            st.info("Yaklaşan olay bulunamadı.")

# Teknik Analiz Sayfası
def technical_analysis_page():
    st.markdown('<h1 class="main-header">📊 Teknik Analiz</h1>', unsafe_allow_html=True)
    
    if not technical_analysis_engine:
        st.error("Teknik analiz motoru kullanılamıyor.")
        return
    
    # Hisse seçimi
    symbol = st.selectbox("Hisse Seçin", ["KCHOL.IS", "THYAO.IS", "GARAN.IS", "AKBNK.IS"])
    
    if st.button("Teknik Analiz Yap", type="primary", key="run_technical_analysis"):
        with st.spinner("Teknik analiz yapılıyor..."):
            result = technical_analysis_engine.process_technical_analysis_request(f"{symbol} teknik analiz yap")
            
            if result.get('error'):
                st.error(f"Teknik analiz hatası: {result['error']}")
            else:
                st.markdown("### 📈 Analiz Sonuçları")
                st.write(result.get('analysis', ''))
                
                st.markdown("### 📋 Özet")
                st.write(result.get('summary', ''))
                
                # Grafikler
                if result.get('charts'):
                    st.markdown("### 📊 Grafikler")
                    for i, chart in enumerate(result['charts']):
                        st.markdown(f"**{chart.get('title', f'Grafik {i+1}')}**")
                        
                        # HTML img tagını işle
                        chart_data = chart.get('data', '')
                        if chart_data.startswith('<img src="data:image/png;base64,'):
                            # Base64 veriyi çıkar
                            import re
                            img_pattern = r'<img src="data:image/png;base64,([^"]+)"[^>]*>'
                            match = re.search(img_pattern, chart_data)
                            if match:
                                base64_data = match.group(1)
                                st.image(f"data:image/png;base64,{base64_data}", 
                                        use_column_width=True)
                            else:
                                st.write(chart_data)
                        else:
                            st.write(chart_data)
                
                # Eğer analysis içinde HTML img tagları varsa onları da işle
                analysis_text = result.get('analysis', '')
                if '<img src="data:image/png;base64,' in analysis_text:
                    # HTML img taglarını bul ve işle
                    import re
                    img_pattern = r'<img src="data:image/png;base64,([^"]+)"[^>]*>'
                    matches = re.findall(img_pattern, analysis_text)
                    
                    if matches:
                        st.markdown("### 📊 Analiz Grafikleri")
                        for base64_data in matches:
                            st.image(f"data:image/png;base64,{base64_data}", 
                                    use_column_width=True)
                    
                    # HTML img taglarını temizle
                    analysis_text = re.sub(img_pattern, '', analysis_text)
                    result['analysis'] = analysis_text

# Alarm Yönetimi Sayfası
def alerts_page():
    st.markdown('<h1 class="main-header">🔔 Alarm Yönetimi</h1>', unsafe_allow_html=True)
    
    if not financial_alert_system:
        st.error("Alarm sistemi kullanılamıyor.")
        return
    
    user_id = f"user_{st.session_state.current_session_id}"
    
    # Alarm oluşturma
    st.markdown("### ➕ Yeni Alarm Oluştur")
    
    with st.form("create_alert_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("Hisse Kodu", placeholder="KCHOL")
        
        with col2:
            event_type = st.selectbox("Olay Türü", ["bilanço", "genel_kurul", "temettü", "diğer"])
        
        with col3:
            days_before = st.number_input("Kaç Gün Önce", min_value=1, max_value=30, value=1)
        
        event_date = st.date_input("Olay Tarihi")
        description = st.text_area("Açıklama")
        
        if st.form_submit_button("Alarm Oluştur", type="primary", key="create_alert_submit"):
            if symbol and event_date and description:
                result = financial_alert_system.create_alert(
                    user_id=user_id,
                    symbol=symbol.upper(),
                    event_type=event_type,
                    event_date=event_date.strftime("%Y-%m-%d"),
                    description=description,
                    days_before=days_before
                )
                
                if result['success']:
                    st.success(result['message'])
                else:
                    st.error(result['error'])
            else:
                st.error("Lütfen tüm alanları doldurun.")
    
    # Mevcut alarmlar
    st.markdown("### 📋 Mevcut Alarmlar")
    
    try:
        active_alerts = financial_alert_system.get_user_alerts(user_id, 'active')
        triggered_alerts = financial_alert_system.get_user_alerts(user_id, 'triggered')
        
        if active_alerts:
            st.markdown("#### 🔔 Aktif Alarmlar")
            for alert in active_alerts:
                with st.expander(f"{alert.symbol} - {alert.event_type} - {alert.event_date}"):
                    st.write(f"**Açıklama:** {alert.description}")
                    st.write(f"**Alarm Tarihi:** {alert.alert_date}")
                    st.write(f"**Oluşturulma:** {alert.created_at}")
                    
                    if st.button(f"İptal Et", key=f"cancel_alert_{alert.id}"):
                        success = financial_alert_system.cancel_alert(alert.id, user_id)
                        if success:
                            st.success("Alarm iptal edildi")
                            st.rerun()
                        else:
                            st.error("Alarm iptal edilemedi")
        
        if triggered_alerts:
            st.markdown("#### ⚡ Tetiklenen Alarmlar")
            for alert in triggered_alerts:
                st.info(f"**{alert.symbol}** - {alert.event_type} - {alert.event_date} (Tetiklenme: {alert.triggered_at})")
        
        if not active_alerts and not triggered_alerts:
            st.info("Henüz alarm bulunmuyor.")
            
    except Exception as e:
        st.error(f"Alarmlar yüklenirken hata oluştu: {str(e)}")

# Ana uygulama
def main():
    # Sayfa seçimi
    if 'page' not in st.session_state:
        st.session_state.page = "Ana Sayfa"
    
    # Sayfa yönlendirme
    if st.session_state.page == "Ana Sayfa":
        main_page()
    elif st.session_state.page == "Portföy Yönetimi":
        portfolio_page()
    elif st.session_state.page == "Finansal Takvim":
        calendar_page()
    elif st.session_state.page == "Teknik Analiz":
        technical_analysis_page()
    elif st.session_state.page == "Alarm Yönetimi":
        alerts_page()

if __name__ == "__main__":
    main()