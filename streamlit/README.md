# SignalLens — Dezenformasyon Analizi Streamlit Dashboard

Bu paket, proje veri setini otomatik yükleyen ve sekiz sayfalık analiz akışını sunan nihai uygulamadır.

- **Temel:** Senin geliştirdiğin gelişmiş "SignalLens" dashboard'u — canlı veri yükleme,
  canlı temizleme pipeline'ı, notebook'un GridSearchCV ile bulduğu gerçek hiperparametrelerle
  canlı model eğitimi, ROC/PR eğrileri, karar eşiği optimizasyonu, özellik önem analizi,
  canlı tahmin demosu ve Google Trends canlı doğrulaması.
- **Eklenen:** Benim ilk sunumumdan tek eksik olan **sızıntı (leakage) kontrolü** —
  dil ve konu-dışı filtrelerinin dezenformasyon sınıfını sistematik olarak hedefleyip
  hedeflemediğinin (çıkarılan kayıtların 'yes' oranı ve kategori dağılımı genel ortalamayla
  karşılaştırılarak) doğrulanması. Bu bölüm artık **"🧹 Veri Temizleme"** sayfasında,
  4 filtreleme adımının hemen ardından yer alıyor.

Bu sürümde metin okunabilirliği artırılmış, veri keşfi ve temizleme sayfaları genişletilmiş,
`sample_weight` model tekrarları kaldırılmış ve optimizasyon yöntemi açıklanmıştır.

## Kurulum ve Çalıştırma

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

`veri_seti_duzenlendi.xlsx` dosyası aynı klasörde olduğu için uygulama açılışta
otomatik olarak yüklenir (sol menüden farklı bir dosya da yükleyebilirsin).

## Sayfalar

1. 🏠 Genel Bakış — proje özeti, vaka çalışmaları
2. 📊 Veri Keşfi — satır/sütun bilgisi, değişken tablosu ve temel EDA grafikleri
3. 🧹 Veri Temizleme — dolu örnek tabloları, sızıntı kontrolü, önce/sonra karşılaştırma ve en altta pipeline
4. ❓ Araştırma Soruları — notebook'taki 9 sorunun tamamı, canlı hesaplanan
5. 🤖 Modelleme & Optimizasyon — performans, GridSearchCV açıklaması, hata analizi, ROC/PR ve eşik optimizasyonu
6. 🔎 Özellik Analizi — model bazında en etkili kelimeler/n-gramlar
7. 🎯 Canlı Tahmin — kendi metnini yaz, anlık sınıflandırma sonucu al
8. 📌 Sonuç — bulguların ve yöntemsel sınırlılıkların özeti

## Notlar

- Model karşılaştırma tablosundaki nihai sayılar (`FIXED_MODEL_RESULTS`), notebook'un
  gerçek GridSearchCV sonrası test sonuçlarına sabitlenmiştir — paket/sürüm farklarından
  etkilenmez. Diğer tüm analizler (confusion matrix, ROC, eşik optimizasyonu, özellik
  önemi, canlı tahmin) aynı hiperparametrelerle **canlı olarak** yeniden eğitilen
  modellerden hesaplanır.
- Google Trends canlı çekimi internet bağlantısı gerektirir; bağlantı yoksa uygulamanın
  geri kalanı sorunsuz çalışmaya devam eder.
- Duygu analizi için `vaderSentiment` (çevrimdışı) veya `nltk` VADER sözlüğü kullanılır;
  ikisi de yoksa bu bölüm nazikçe devre dışı kalır.
