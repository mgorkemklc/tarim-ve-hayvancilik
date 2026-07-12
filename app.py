import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Veri Analiz Platformu", layout="wide")

st.title("🌍 Evrensel AutoML (Otomatik Makine Öğrenimi) Kokpiti")
st.markdown("Herhangi bir veri setini yükleyin, kolonları eşleştirin, hedef tarihi belirleyin ve 5 farklı algoritmayı anında yarıştırın.")
st.markdown("---")

# --- HTML/CSS TAKVİM OLUŞTURUCU (Boşluk/Markdown hatası giderildi) ---
def render_calendar_html(forecast_series):
    # Markdown'un kodu "kod bloğu" sanmaması için başındaki boşluklar tamamen kaldırıldı
    html = "<style>\n"
    html += ".calendar-container { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 15px; margin-bottom: 30px; }\n"
    html += ".calendar-box { border: 1px solid #e0e0e0; border-radius: 10px; padding: 12px; width: 110px; text-align: center; background-color: #ffffff; box-shadow: 2px 4px 10px rgba(0,0,0,0.05); transition: all 0.3s ease; cursor: default; }\n"
    html += ".calendar-box:hover { transform: translateY(-5px); border-color: #004a8b; box-shadow: 2px 8px 15px rgba(0, 74, 139, 0.2); }\n"
    html += ".cal-day { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; }\n"
    html += ".cal-date { font-size: 16px; font-weight: bold; color: #333; margin: 5px 0; }\n"
    html += ".cal-val { font-size: 19px; color: #28a745; font-weight: 900; }\n"
    html += "</style>\n"
    html += "<div class='calendar-container'>\n"
    
    tr_days = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    tr_months = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]

    for date, val in forecast_series.items():
        day_str = tr_days[date.weekday()]
        month_str = tr_months[date.month - 1]
        date_str = f"{date.day} {month_str}"
        val_fmt = f"{val:.1f}"

        html += f"<div class='calendar-box'>\n"
        html += f"<div class='cal-day'>{day_str}</div>\n"
        html += f"<div class='cal-date'>{date_str}</div>\n"
        html += f"<div class='cal-val'>{val_fmt}</div>\n"
        html += f"</div>\n"
        
    html += "</div>"
    return html

# --- 5 ALGORİTMAYI EĞİTEN ANA MOTOR ---
def advanced_automl_engine(df_ts, target_date):
    logs = []
    logs.append("⚙️ **AutoML Süreci Başlatıldı:** Zaman serisi günlük (D) periyoda uyarlandı.")
    
    train = df_ts.iloc[:-7]
    test = df_ts.iloc[-7:]
    logs.append(f"📦 Veri seti bölündü: {len(train)} gün eğitim, {len(test)} gün test verisi.")
    
    df_ts_ml = df_ts.copy().reset_index()
    df_ts_ml['dayofweek'] = df_ts_ml['tarih'].dt.dayofweek
    df_ts_ml['month'] = df_ts_ml['tarih'].dt.month
    df_ts_ml['day'] = df_ts_ml['tarih'].dt.day
    
    X_train = df_ts_ml.iloc[:-7][['dayofweek', 'month', 'day']]
    y_train = train['hedef_degisken']
    X_test = df_ts_ml.iloc[-7:][['dayofweek', 'month', 'day']]
    y_test = test['hedef_degisken']
    
    ml_models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(C=1.0, epsilon=0.2)
    }
    
    results = {}
    
    try:
        logs.append("⏳ 1/5: **Holt-Winters** modeli eğitiliyor...")
        hw = ExponentialSmoothing(train['hedef_degisken'], trend='add', seasonal='add', seasonal_periods=7).fit()
        hw_pred = hw.forecast(len(test))
        mape = mean_absolute_percentage_error(test['hedef_degisken'], hw_pred)
        results['Holt-Winters'] = mape
        logs.append(f"✅ **Holt-Winters** tamamlandı. Hata Oranı: %{mape*100:.2f}")
    except Exception as e:
        results['Holt-Winters'] = 9.99
        logs.append(f"❌ **Holt-Winters** hata verdi: {str(e)}")
        
    idx = 2
    for name, model in ml_models.items():
        try:
            logs.append(f"⏳ {idx}/5: **{name}** modeli eğitiliyor...")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, pred)
            results[name] = mape
            logs.append(f"✅ **{name}** tamamlandı. Hata Oranı: %{mape*100:.2f}")
        except Exception as e:
            results[name] = 9.99
            logs.append(f"❌ **{name}** başarısız oldu.")
        idx += 1
        
    best_model_name = min(results, key=results.get)
    best_mape = results[best_model_name]
    best_accuracy_pct = max(0, (1 - best_mape) * 100)
    
    logs.append(f"🏆 **Kazanan Algoritma:** En düşük hata oranıyla **{best_model_name}** seçildi!")
    
    last_date = df_ts.index[-1].date()
    days_to_target = (target_date - last_date).days
    
    forecast_days = max(30, days_to_target + 5) if days_to_target > 0 else 30
    future_dates = pd.date_range(start=df_ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    
    if best_model_name == 'Holt-Winters':
        final_model = ExponentialSmoothing(df_ts['hedef_degisken'], trend='add', seasonal='add', seasonal_periods=7).fit()
        forecast = final_model.forecast(forecast_days)
    else:
        X_full = df_ts_ml[['dayofweek', 'month', 'day']]
        y_full = df_ts['hedef_degisken']
        final_model = ml_models[best_model_name].fit(X_full, y_full)
        
        future_df = pd.DataFrame({'tarih': future_dates})
        future_df['dayofweek'] = future_df['tarih'].dt.dayofweek
        future_df['month'] = future_df['tarih'].dt.month
        future_df['day'] = future_df['tarih'].dt.day
        
        pred_values = final_model.predict(future_df[['dayofweek', 'month', 'day']])
        forecast = pd.Series(pred_values, index=future_dates)
        
    target_date_ts = pd.to_datetime(target_date)
    
    if target_date_ts in forecast.index:
        target_prediction = forecast.loc[target_date_ts]
    elif target_date_ts in df_ts.index:
        target_prediction = df_ts.loc[target_date_ts]
        logs.append(f"ℹ️ Seçtiğiniz tarih geçmişte kalıyor. Gerçekleşen veri ekrana yansıtıldı.")
    else:
        target_prediction = forecast.mean()
        
    return results, best_model_name, best_accuracy_pct, forecast, logs, target_prediction, forecast_days

# --- SIDEBAR (DİNAMİK VERİ YÜKLEME VE TAKVİM) ---
st.sidebar.header("📂 Veri Yükleme")
yuklenen_dosya = st.sidebar.file_uploader("Herhangi bir veri seti yükleyin (.csv veya .xlsx)", type=['csv', 'xlsx'])

if yuklenen_dosya is not None:
    try:
        if yuklenen_dosya.name.endswith('.csv'):
            df = pd.read_csv(yuklenen_dosya)
        else:
            df = pd.read_excel(yuklenen_dosya)
            
        st.sidebar.success("✅ Veri okundu! Eşleştirmeleri yapın:")
        kolonlar = df.columns.tolist()
        
        tarih_kolonu = st.sidebar.selectbox("📅 Tarih (Date) Kolonu:", kolonlar)
        hedef_kolon = st.sidebar.selectbox("🎯 Tahmin Edilecek Değer (Target):", kolonlar)
        kategori_kolonu = st.sidebar.selectbox("🏷️ Kategori / İsim Kolonu:", kolonlar)
        
        st.sidebar.markdown("---")
        hedef_tarih = st.sidebar.date_input(
            "🔮 Tahmin İçin Gelecek Bir Tarih Seçin:", 
            value=datetime.date.today() + datetime.timedelta(days=14)
        )
        
        if st.sidebar.button("🚀 Yapay Zekayı Başlat"):
            st.header(f"📊 Analiz Raporu: {yuklenen_dosya.name}")
            
            df = df.rename(columns={tarih_kolonu: 'tarih', hedef_kolon: 'hedef_degisken', kategori_kolonu: 'kategori'})
            
            df['tarih'] = pd.to_datetime(df['tarih'], errors='coerce')
            df['hedef_degisken'] = pd.to_numeric(df['hedef_degisken'], errors='coerce')
            df = df.dropna(subset=['tarih', 'hedef_degisken']) 
            
            secili_isim = df['kategori'].value_counts().index[0]
            df_filtre = df[df['kategori'] == secili_isim].copy()
            df_filtre = df_filtre.sort_values('tarih')
            
            df_ts = df_filtre.groupby('tarih')['hedef_degisken'].mean().reset_index().set_index('tarih')
            df_ts = df_ts.asfreq('D').ffill()
            
            if len(df_ts) < 15:
                st.error(f"⚠️ HATA: Seçilen birim ({secili_isim}) için yeterli tarihsel veri yok. En az 15 gün gerekli.")
            else:
                with st.spinner("🤖 Modeller eğitiliyor ve birbiriyle yarışıyor..."):
                    results, best_model, accuracy, forecast, logs, target_prediction, forecast_days = advanced_automl_engine(df_ts, hedef_tarih)
                
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Analiz Edilen Birim", value=str(secili_isim))
                m2.metric(label="🏆 Kazanan Model (Doğruluk)", value=f"{best_model} (% {accuracy:.2f})")
                m3.metric(label=f"🎯 {hedef_tarih.strftime('%d.%m.%Y')} Tahmini", value=f"{target_prediction:.1f}")
                
                with st.expander("🕵️ Yapay Zeka Arka Planda Hangi Adımları İzledi?"):
                    for log in logs:
                        st.write(log)
                        
                g1, g2 = st.columns(2)
                
                with g1:
                    st.subheader(f"📈 Gelecek Projeksiyon Grafiği")
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    ax.plot(df_ts.index[-45:], df_ts['hedef_degisken'].iloc[-45:], label='Gerçek Veri', color='#004a8b', linewidth=2)
                    ax.plot(forecast.index, forecast, label=f'Yapay Zeka ({best_model})', color='#ff4b4b', linestyle='--', linewidth=2)
                    
                    target_date_ts = pd.to_datetime(hedef_tarih)
                    if target_date_ts in forecast.index:
                        ax.scatter([target_date_ts], [target_prediction], color='gold', s=120, zorder=5, label='Seçilen Tarih')
                        ax.axvline(x=target_date_ts, color='gray', linestyle=':', alpha=0.7)
                        
                    ax.set_title(f"{secili_isim} Verisi Zaman Serisi")
                    ax.legend()
                    st.pyplot(fig)
                    
                with g2:
                    st.subheader("⚔️ Modellerin Hata (MAPE) Yarışı")
                    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                    model_names = list(results.keys())
                    error_rates = [v * 100 for v in results.values()]
                    colors = ['#28a745' if name == best_model else '#adb5bd' for name in model_names]
                    sns.barplot(x=error_rates, y=model_names, palette=colors, ax=ax2)
                    ax2.set_xlabel("Hata Oranı (% - Düşük Olan İyidir)")
                    st.pyplot(fig2)

                # --- TAKVİM BÖLÜMÜ ---
                st.markdown("---")
                st.subheader(f"📅 Günlük Tahmin Takvimi (Son Veriden {hedef_tarih.strftime('%d.%m.%Y')} Tarihine Kadar)")
                st.markdown("Seçtiğiniz hedefe kadar olan her gün için yapay zekanın ürettiği nokta atışı tahminler:")
                
                cal_forecast = forecast[forecast.index <= pd.to_datetime(hedef_tarih)]
                
                if len(cal_forecast) > 0:
                    st.markdown(render_calendar_html(cal_forecast), unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Seçilen tarih geçmişte kaldığı için gelecek takvimi oluşturulamadı. Lütfen ileri bir tarih seçin.")

    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)} \nLütfen eşleştirdiğiniz kolonların doğru formata sahip olduğundan emin olun.")
else:
    st.info("👈 Lütfen sol menüden Excel veya CSV formatında bir veri yükleyin.")