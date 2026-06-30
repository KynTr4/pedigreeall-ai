# AT Yarış Tahmini — A–Z Teknik Denetim ve Kök Neden Analizi

**Denetim tarihi:** 2026-06-30 (Europe/Istanbul)  
**Kapsam:** Mevcut `at_yaris_tahmini` çalışma alanı; ayrı proje oluşturulmadı.  
**Yöntem:** Kaynak kod, 2.9 GB SQLite veritabanı, model artifact'ları, CSV/Parquet çıktıları, raporlar, loglar, systemd tanımları ve testler salt okunur incelendi. Model dosyaları değiştirilmedi; yalnız bu rapor eklendi.

## 1. Yönetici özeti

Proje veri toplama, immutable snapshot şeması, read-only dashboard, sistem servisleri ve test altyapısı açısından ciddi miktarda mühendislik içeriyor. Buna rağmen mevcut model performans iddiaları bilimsel olarak geçerli değil. İki bloklayıcı kök neden vardır:

1. **`handicap_rating` aynı yarışın sonucundan sonra güncellenmiş bilgiyi taşıyor.** 2026'da kazananların yarış satırındaki rating değişimi ortalama `+6.98`, ikincilerin `+2.62`. Mevcut rating ile yarış içindeki en yüksek rating tek başına `1696/2452 = %69.22` Top-1 yaparken yalnız önceki yarıştan bilinen rating `568/2284 = %24.88` yapıyor. Aynı CatBoost kurulumu mevcut rating ile `%58.77`, bir yarış geciktirilmiş rating ile `%29.40` yaptı. Bu doğrudan olmasa da güçlü bir **outcome leakage** kanıtıdır.
2. **Tarihsel veri tam yarış programı değil, keşfedilmiş atların geçmiş satırlarının birleşimi.** 2024'te 5.336 yarışın yalnız 2.229'unda veri kümesi içinde tam bir kazanan ve en az iki koşucu var. 1.266 yarış tek atlı, 1.840 yarışta kazanan veri kümesinde yok. Backtest kazananın veri kümesinde bulunduğu yarışları seçiyor ve eksik rakipler arasında sıralama yapıyor. Bu hedefe bağlı seçim yanlılığıdır.

Bu iki sorun nedeniyle:

- Raporlanan 2026 CatBoost Top-1 `%57.95` canlı beklenti olarak kullanılamaz.
- Raporlanan CatBoost Top-1 ROI `%125.33` sertifikalı değildir.
- Calibration, SHAP ve feature importance çıktıları teknik olarak hesaplanmış olsa da sızıntılı/yanlı değerlendirme evrenini açıklamaktadır.
- Sistem **production ready değildir**; mevcut doğru konum **shadow altyapısı kurulmuş fakat henüz canlı gözlemi olmayan deneysel sistem**dir.

**Genel olgunluk puanı: 3/10.** Altyapı mühendisliği yaklaşık 5/10, model/veri bilimi geçerliliği 2/10, canlı doğrulama olgunluğu 1/10 seviyesindedir.

## 2. Doğrulanmış mevcut durum

| Alan | Bulgular | Durum |
|---|---:|---|
| SQLite bütünlüğü | `PRAGMA quick_check = ok` | PASS |
| `horse_races` | 961.695 satır, 144.850 yarış, 49.280 at, 1970-01-01–2026-06-26 | Bilgi |
| Final CSV | 961.695 satır × 62 kolon | PASS |
| Final Parquet | 961.695 satır × 62 kolon | PASS |
| CSV/Parquet kolon sırası | Aynı | PASS |
| Duplicate `(horse_id,race_id)` | 0 | PASS |
| Unit/integration testleri | 72 PASS, 1 deprecation warning | PASS |
| Python derleme kontrolü | `compileall` PASS | PASS |
| Program snapshot | 589 satır / 64 yarış | Çok düşük kapsam |
| Yarış öncesi program snapshot | 84/589 satır / 8 yarış | FAIL — kapsama göre |
| Yarış sonrası/geç program capture | 505/589 satır | Risk |
| AGF snapshot | 585 satır; yalnız 82'si kendi yarışından önce | Çok düşük kapsam |
| Odds snapshot | 539 satır; yalnız 72'si kendi yarışından önce | Çok düşük kapsam |
| `race_results` | 0 | Canlı sonuç yok |
| `prediction_snapshots` | 0 | Canlı tahmin yok |
| `prediction_results` | 0 | Canlı değerlendirme yok |
| Shadow günleri | 0/90 | Production ready değil |

## 3. En kritik 10 sorun

### P0-1 — Post-race `handicap_rating` leakage

**Kanıt:**

- `build_final_dataset.py` tarihsel `horse_races.rating` alanını doğrudan `handicap_rating` yapıyor.
- Bu tarihsel satırlarda yarış zamanında alınmış `captured_at` yok.
- 2026 geçerli yarışlarında rating değişimi sonuç sırasıyla monoton biçimde ilişkili: kazanan `+6.98`, ikinci `+2.62`, üçüncü `+1.03`, altıncı `-0.28` ortalama.
- En yüksek mevcut rating baseline'ı `%69.22`; en yüksek önceki rating `%24.88`.
- Backtest feature importance raporunda CatBoost için `handicap_rating`: gain `37.08`, permutation importance `0.40576`; açık ara baskın feature.
- Bağımsız hassasiyet deneyi: aynı CatBoost, mevcut rating ile `1441/2452 = %58.77`; lagged rating ile `721/2452 = %29.40`.

**Kök neden:** Alan adı ve şema kontrolüyle “sonuç kolonu değil” denmiş; fakat semantik provenance kontrol edilmemiş. Sonucun kendisini veya sonuç sonrası rating güncellemesini taşıyan dolaylı feature yakalanmamış.

**Etki:** Top-1 skorunda yaklaşık `29.4 yüzde puan` iyimserlik görüldü. Bu, kesin üretim modeli etkisi değil; aynı mimaride kontrollü hassasiyet ölçümüdür. Yine de mevcut performans iddiasını geçersiz kılacak büyüklüktedir.

**Güven:** Çok yüksek.

### P0-2 — Eksik yarış alanı ve hedefe bağlı yarış seçimi

| Yıl | Program/DB yarış | Geçerli backtest yarışı | Tek atlı | Kazanan yok | Çoklu kazanan |
|---:|---:|---:|---:|---:|---:|
| 2024 | 5.336 | 2.229 | 1.266 | 1.840 | 1 |
| 2025 | 6.320 | 4.415 | 473 | 1.427 | 5 |
| 2026 | 2.930 | 2.452 | 34 | 440 | 4 |

`horse_races`, yarış programından bütün starter'ları toplayan bir fact table değil; keşfedilmiş atların geçmişlerinin birleşimi. Backtest `tam bir kazanan + en az iki satır` koşuluyla yarış seçiyor. Böylece gerçek kazananın yakalanmış olması değerlendirmeye giriş şartına dönüşüyor, yakalanmamış rakipler ise yarıştan çıkarılıyor.

**Etki:** Top-1, Top-N, calibration ve ROI'nin tamamı iyimser ve temsil gücü düşük olabilir. 2026 holdout'ta 2.930 yarışın tamamı değil `2.452 (%83.69)` yarış değerlendirilmiştir.

**Güven:** Çok yüksek.

### P0-3 — Tarihsel backtest as-of sertifikalı değil

`reports/leakage_gate_v2.md` açıkça PASS sonucunun yalnız `output/asof_features.parquet` için geçerli olduğunu söylüyor. Bu dosya yalnız 84 at satırı ve 8 yarış içeriyor. `%57.95` raporlanan backtest ise `output/final_benter_dataset.parquet` içindeki tarihsel, timestamp'siz 961.695 satırdan geliyor.

Dolayısıyla şu iki ifade birbirine karıştırılmamalıdır:

- “As-of builder kabul ettiği satırlarda `captured_at < race_start_at` uyguluyor.” — Doğru.
- “Raporlanan tarihsel model performansı as-of leakage-safe.” — Kanıtlanmamış ve rating bulgusuyla yanlışlanmış.

**Güven:** Çok yüksek.

### P0-4 — Production XGBoost artifact'ı tarih ayrıştırma hatasıyla eğitilmiş

`train_xgboost_production.py:119`, kaynak format `DD.MM.YYYY` olmasına rağmen `pd.to_datetime(..., errors="coerce")` çağrısında `dayfirst=True` kullanmıyor. Üretilen raporun kendisi anomali gösteriyor:

- Eğitim: 1979-03-31–2010-04-02
- Test: 2010-04-03–**2026-12-06**
- Artifact üretim tarihi: 2026-06-26
- Top-1: `%16.03`

Gelecek tarih 2026-12-06, `06.12.2026` ile `12.06.2026` karışmasının göstergesidir. Shadow ensemble bu `xgboost_production.pkl` dosyasını kullanıyor; backtest'teki `%55.59` XGBoost ise fold içinde yeniden eğitilen başka bir modeldir.

**Etki:** Canlı ensemble üç eşit ağırlıklı bileşenin birinde hatalı tarih evreni ve uyumsuz model vintage'ı kullanıyor.

**Güven:** Çok yüksek.

### P0-5 — Canlı doğrulama veri olarak başlamamış

Ana DB'de:

- `prediction_snapshots = 0`
- `race_results = 0`
- `prediction_results = 0`
- `prediction_feature_snapshots = 0`
- `race_prediction_lifecycle = 0`

Yedi `shadow_monitoring_runs` kaydı var fakat hepsi `SHADOW_WARMUP`, `production_ready=0`; tamamlanan shadow günü `0/90`. Bunlar model performans gözlemi değil, monitörün veri yokken çalıştırılmasıdır.

**Etki:** Canlı accuracy, calibration, drift, ROI, AGF karşılaştırması ve diagnostics için istatistiksel veri yoktur.

**Güven:** Çok yüksek.

### P1-6 — Canlı feature frame eğitim dağılımının dışında ve tarihsel feature'lar boş

`output/asof_features.parquet`:

- 84 satır, 8 yarış;
- yalnız Belmont, Monmouth Park ve San Isidro;
- üç pistin tamamı eğitim kategorilerinde görülmemiş;
- sekiz `race_class` değerinin tamamı eğitimde görülmemiş;
- 20 model feature'ının 11'i `%100` boş: gün farkı, son 3/5/10, dört win-rate, jokey/antrenör oranları, weight/distance change;
- `class_change` ve `surface_change` cold-start'ta `0/1` ile “bilinmiyor” yerine gerçek değişim gibi kodlanıyor.

Direct feature drift göstergeleri de çok yüksek: distance PSI yaklaşık `7.26`, carried weight `5.41`, handicap rating `4.59`. Örnek yalnız 84 satır olduğu için bunlar kesin population drift tahmini değil, bariz domain mismatch alarmıdır.

**Etki:** Model Türkiye tarihsel evreninden tamamen görülmemiş yabancı yarış evrenine, temel form feature'ları median-imputed halde uygulanıyor.

**Güven:** Yüksek.

### P1-7 — Leakage/coverage gate boş veriyle PASS olabiliyor

`shadow_monitor.snapshot_coverage_pass()` history boşsa `archive_integrity=True` kabul ediyor. Prediction tablosu boş olduğunda referans bütünlük sorguları da doğal olarak sıfır hata döndürüyor. Validator, as-of dosyasının non-empty olmasını kontrol ediyor fakat beklenen günlük program yarışlarına karşı zorunlu coverage eşiği uygulamıyor. Bu nedenle:

- 589 program satırının 505'i geç capture olsa da;
- yalnız 8 yarış feature frame'e girse de;
- hiç tahmin üretilmemiş olsa da;

health dashboard “Snapshot Coverage: PASS” gösterebiliyor.

Synthetic prefix/future/target mutation testleri yararlıdır fakat tarihsel gerçek kaynağın semantiğini veya günlük yarış completeness'ını kanıtlamaz.

**Güven:** Çok yüksek.

### P1-8 — ROI sertifikasız ve leakage ile şişmiş

Tarihsel ROI, `horse_races.odds` sonucuna ait final GNY/odds alanını kullanıyor; bunun bahis anında mevcut olduğunu kanıtlayan timestamp yok. Ayrıca:

- post-race rating leakage;
- eksik rakipler ve hedefe bağlı yarış seçimi;
- komisyon, limit, slippage, dead heat, kesinti ve scratch modelinin olmaması

ROI'yi bozuyor. `%125.33` Top-1 ROI ekonomik olarak olağan dışı ve aynı veri kusurlarıyla açıklanabilir. Canlı raporun `ROI = NOT CERTIFIED` demesi doğrudur; tarihsel rapor başlığı/özetinde de aynı sertlikte uyarı gerekir.

**Güven:** Çok yüksek.

### P1-9 — Model artifact provenance ve ensemble tutarsız

Shadow mode şu üç artifact'ı eşit ağırlıkla birleştiriyor:

- `benter_baseline_logistic.pkl`
- `benter_baseline_catboost.pkl`
- `xgboost_production.pkl`

CatBoost ve Logistic için tekrar üretilebilir eğitim manifesti, exact dataset hash'i, split, kod commit'i ve bağımsız evaluation raporu artifact yanında yok. XGBoost farklı tarihte ve hatalı parser ile yeniden eğitilmiş. Backtest fold modelleri dosyaya kaydedilmediği için raporlanan “kazanan CatBoost” shadow'daki CatBoost artifact'ıyla aynı model değildir.

Eşit ağırlıklı ensemble için OOF optimizasyonu veya calibration-ağırlık kanıtı yok. Üstelik holdout'ta CatBoost `%57.95`, ensemble `%56.89`; ensemble iyileştirmiyor.

**Güven:** Yüksek.

### P1-10 — Reproducibility, operasyon ve güvenlik yönetişimi zayıf

- Git'e göre çalışma alanındaki dosyaların neredeyse tamamı untracked; güvenilir commit geçmişi ve değişiklik kaynağı yok.
- `requirements.txt` exact lock değil, geniş sürüm aralıkları kullanıyor.
- CI workflow artifact'ı görülmedi; 72 test yalnız yerel koşuda doğrulandı.
- 2026-06-29 tarihli VPS raporunda web ve results service başarılı görünürken `at-yaris-live-results.timer` için next run `n/a` ve timer inactive görünmektedir. Bu tarihsel kanıttır; bugünkü VPS durumu erişim olmadığı için doğrulanamadı.
- Aynı VPS doğrulama raporunda Basic Auth parolası komut satırı içinde açık biçimde arşivlenmiş. Değer bu raporda tekrarlanmamıştır; credential rotate edilmelidir.
- Çok sayıda geniş `except Exception`, çıplak `except` ve sessiz `pass` vardır. Özellikle ingestion kodunda veri kaybını teknik başarı gibi gösterebilir.
- Dashboard SQLite bağlantısında `mode=ro` ve `PRAGMA query_only=ON` kullanılması olumlu; fakat 50.000+ gerçek prediction üzerinde `<500 ms` benchmark'ı yapılamadı, çünkü ana DB'de prediction yok.

**Güven:** Yüksek.

## 4. Veri mimarisi denetimi

### 4.1 Güçlü taraflar

- Snapshot tablolarında update/delete engelleyen append-only trigger'lar var.
- `source_request_id`, `captured_at`, `race_start_at` ve feature hash alanları tasarımda mevcut.
- As-of join `MAX(captured_at) WHERE captured_at < race_start_at` ilkesini uyguluyor.
- Sonuçlar prediction'lardan ayrı tablolarda tutuluyor.
- Dashboard `mode=ro` ve `query_only` bağlantı kullanıyor.
- Final CSV/Parquet aynı 62 kolon ve aynı 961.695 parse edilmiş satırı içeriyor.

### 4.2 Zayıf taraflar

- Snapshot sistemi yalnız tek capture anına yakın 64 yarışla doldurulmuş; tarihsel training evrenini kapsamıyor.
- Program snapshot'larının `%85.74`ü (`505/589`) yarıştan sonra alınmış.
- `race_results=0` olduğu için rolling canlı feature'ların tamamı boş.
- Program kimlikleri (`prog_...`, `name:<hash>`, `tjk:<id>`) ile tarihsel (`race_id` sayısal, `horse:<id>`) evrenleri arasında doğrulanmış kalıcı identity bridge yok.
- Sonuç coverage'ın yerel DB ve VPS kopyası arasında farklı olması environment/data drift göstergesidir.

## 5. Feature provenance matrisi

| Feature grubu | Kaynak | Canlı as-of durumu | Tarihsel backtest durumu | Karar |
|---|---|---|---|---|
| track, distance, surface, race_class, weight, draw | `program_snapshots` | Capture yarıştan önceyse güvenli | `horse_races` post-hoc indirme; timestamp yok | Canlı kabul, tarihsel kanıtsız |
| handicap_rating | program / `horse_races.rating` | Snapshot öncesiyse olası güvenli | Aynı sonuç sonrası güncelleme taşıyor | Tarihsel modelden çıkar/lagle |
| days_since_last_race | önceki `race_results` | Tasarım güvenli, veri yok | date-only geçmiş | Canlı coverage gerekli |
| last_3/5/10_avg_position | önceki sonuçlar | Shift/as-of tasarımı güvenli, veri yok | Shift doğru; kaynak timestamp yok | Rebuild gerekli |
| surface/distance/track win rate | önceki sonuçlar | Tasarım güvenli, veri yok | Prior cumulative | Rebuild gerekli |
| jockey/trainer-horse win rate | program + önceki sonuç | Tasarım güvenli, veri yok | Current jockey/trainer'ın as-of zamanı kanıtsız | Provenance gerekli |
| weight/class/distance/surface change | current program + prior | Cold-start kodlaması sorunlu | Date order kullanıyor | Unknown flag eklenmeli |
| AGF, odds | snapshot tabloları | Feature contract dışında | AGF tamamen boş, odds result alanı | Modelden ayrı tutmak doğru |

Yaş, cinsiyet, pedigree, pace/speed, class-normalized time, form cycle, layoff nonlinearity, jockey/trainer güncel formu ve tam yarış koşulları model sözleşmesinde yoktur. Bunları eklemek leakage problemi çözülmeden öncelik olmamalıdır.

## 6. Backtest ve metrik denetimi

### 6.1 Holdout'ın tamamı değerlendirildi mi?

Hayır. DB'de 2026 için 2.930 yarış var. Backtest yalnız tam bir kazanan ve en az iki satırı bulunan 2.452 yarışı değerlendiriyor:

`2452 / 2930 = %83.69`

Kalan 478 yarış: 34 tek satırlı, 440 kazananı bulunmayan, 4 birden fazla kazanan görünen yarıştır.

### 6.2 Top-1 tam olarak nasıl hesaplanıyor?

Her evaluation fold içinde her modelin ham olasılığı yarış içindeki toplamına bölünerek normalize edilir. Sonra her `race_id` için en yüksek normalize olasılıklı at rank 1 seçilir. Top-1:

`rank=1 seçilen ve finish_position=1 olan yarış sayısı / değerlendirilen yarış sayısı`

2026 CatBoost için:

`1421 / 2452 = %57.9527`

Bu hesap aritmetik olarak doğru yeniden üretildi. Sorun formülde değil, değerlendirmeye giren yarış/feature evrenindedir.

### 6.3 Karşılaştırmalı baseline'lar

| 2026 valid-race yöntem | Yarış | Top-1 |
|---|---:|---:|
| Final odds favorisi | 2.452 | %38.56 |
| En yüksek mevcut `handicap_rating` | 2.452 | %69.22 |
| En yüksek önceki yarış rating'i | 2.284 | %24.88 |
| En iyi last-3 ortalaması | 2.346 | %27.11 |
| Raporlanan CatBoost | 2.452 | %57.95 |
| Raporlanan Ensemble | 2.452 | %56.89 |
| Kontrollü CatBoost + lagged rating | 2.452 | %29.40 |

Bu tablo rating leakage'in model skorunu açıklayan ana mekanizma olduğunu gösterir.

### 6.4 “Accuracy” isimlendirmesi

`model_scores_v2.csv` içindeki yaklaşık `%90` `accuracy`, at satırı bazında 0/1 sınıflandırma accuracy'sidir ve negatif sınıf baskın olduğu için yarış tahmini başarısı değildir. Kullanıcıya sunulacak ana metrik Top-1/Top-N olmalıdır.

## 7. Model ve explainability denetimi

- CatBoost mevcut sızıntılı holdout'ta en iyi modeldir; ensemble onu geçmiyor.
- Logistic Top-1 yaklaşık `%34.75`; ağaç modellerindeki rating etkisinin altında kalıyor.
- Backtest SHAP çıktısı mevcut ve `handicap_rating` baskınlığını doğruluyor. Bu “model neden kazandı?” sorusuna cevap veriyor ama sebep istenmeyen post-race bilgidir.
- Canlı diagnostics'in archived SHAP yoksa “Not available” demesi doğru; read-only sayfada modeli yeniden çalıştırmıyor.
- Ana DB'de prediction olmadığı için yarış bazlı SHAP/feature karşılaştırması fiilen gösterilemez.
- Artifact hash'leri üretiliyor fakat hash tek başına eğitim provenance'ı değildir; dataset hash + kod commit + split manifest gerekir.

## 8. AGF ve odds denetimi

- Tarihsel final dataset'te AGF `%100` boş; AGF favori karşılaştırması yapılamaz.
- AGF snapshot'ta 585 satır var, yalnız 82 satır ilgili yarıştan önce.
- Odds snapshot'ta 539 satır var, yalnız 72 satır ilgili yarıştan önce.
- Shadow prediction arşivlemesi prediction zamanından önceki en son AGF/odds'u seçiyor; tasarım doğru.
- Ancak snapshot hacmi canlı performans veya ROI sertifikası için yetersiz.
- Official/result odds ile pre-race odds birbirinden ayrılmış; canlı ROI'nin `NOT CERTIFIED` olması doğru davranış.

## 9. Pipeline, scheduler ve dashboard denetimi

### Olumlu

- Daily, AGF, race-freeze, live-results, results-update, backup ve web için ayrı systemd unit'leri var.
- Runner lock PID/hostname metadata ve stale lock recovery içeriyor.
- Active runner durumunda skip + exit 0 politikası uygulanmış.
- Web API salt okunur.
- Race-day, performance, diagnostics ve bet simulator endpoint'leri pagination/export içeriyor.

### Riskler

- 2026-06-29 deploy kanıtında live-results timer inactive görünmüş; bugünkü durum kanıtlanmadı.
- Daily pipeline prediction üretmiyor; prediction race-freeze timer'ın doğru pencereye denk gelmesine bağımlı.
- Yerel DB'nin 26 Haziran snapshot'ından sonra ilerlememesi, otomasyonun bu çalışma alanında devam etmediğini gösteriyor.
- `Snapshot Coverage PASS`, günlük beklenen yarış kapsamından ziyade var olan archive satırlarının tutarlılığını ölçüyor.
- Performance/bet SQL'si Top-1 seçimini doğru yapıyor; fakat official sonuç odds'u pre-race odds yerine kullanarak ekonomik simülasyonu sertifikasız bırakıyor.
- 50k kayıt performans şartı gerçek production hacminde benchmark edilmemiştir.

## 10. Test kalitesi

72 testin tamamı geçti. Testler dashboard read-only bağlantısı, lock davranışı, snapshot as-of kuralları, race-day görünümü, diagnostics ve freeze pencereleri için iyi regression koruması sağlıyor.

Eksik kritik testler:

1. `handicap_rating` hedef mutasyonu / sonraki API refresh invariance testi.
2. Tam starter coverage ve gerçek field-size karşılaştırması.
3. “Kazanan veri kümesinde yoksa backtest build FAIL” testi.
4. Production artifact eğitim manifesti ve tarih parser sözleşmesi.
5. Empty prediction history'nin coverage PASS verememesi.
6. 50k+ gerçekçi SQLite benchmark.
7. Timer'ların gerçek VPS üzerinde ardışık günler çalıştığını doğrulayan acceptance testi.
8. Secret scanning.

## 11. Önerilen mimari karar

Mevcut snapshot/dashboard mimarisi korunabilir. Modeli iyileştirmeden önce training fact table yeniden tanımlanmalıdır:

1. Birincil evren `race_program` olmalı: her yarışın bütün starter'ları, scratch durumu ve start zamanı.
2. Her starter satırı için yalnız yarıştan önce bilinen snapshot seçilmeli.
3. Outcome ayrı result fact'ten yarış sonrasında bağlanmalı.
4. Tarihsel as-of kanıtı olmayan alan karantinaya alınmalı; özellikle rating bir yarış geciktirilmeli veya güvenilir pre-race rating snapshot'ı bulunmalı.
5. Yarış completeness gate: beklenen starter sayısı ile feature/result starter sayısı eşit değilse training/evaluation'a alınmamalı ve neden raporlanmalı.
6. Eğitim, validation ve canlı scoring aynı feature builder ve aynı feature contract'ı kullanmalı.

## 12. Önceliklendirilmiş eylem planı

### Kısa vade — bloklayıcılar

1. Mevcut backtest/ROI/model comparison raporlarını **deprecated / not certified** işaretle.
2. `handicap_rating` tarihsel provenance'ını field-semantics seviyesinde çöz; çözülene kadar current-race rating'i feature'dan çıkar veya yalnız lagged kullan.
3. Program-bazlı tam starter fact table üret ve race completeness gate ekle.
4. `train_xgboost_production.py` tarih parser'ını `dayfirst=True` sözleşmesine bağla; mevcut XGB artifact'ını deprecated yap.
5. Empty-history snapshot coverage'ın PASS vermesini engelle.
6. Basic Auth credential'ını rotate et ve raporlardaki açık komutu temizle.

### Orta vade — yeniden doğrulama

7. 2024/2025/2026 dataset'ini yalnız as-of/pre-race kanıtlı feature'larla yeniden kur.
8. Sabit, tam yarış evreninde Logistic/CatBoost/XGBoost'u aynı split ve aynı manifest ile yeniden eğit.
9. Odds favorisi, AGF favorisi, rating baseline ve random/field-size baseline'ları aynı yarışlarda raporla.
10. Calibration ve ROI'yi yalnız leakage-safe holdout'ta yeniden hesapla; ROI için pre-race odds coverage zorunlu olsun.
11. Model artifact manifestine dataset SHA-256, Git commit, Python/dependency lock, feature contract, train cutoff ve metrikleri ekle.
12. Türkiye ve yabancı pistleri ayrı model/domain olarak ele al; görülmemiş ülke/pistte skor üretme veya açık OOD statüsü göster.

### Uzun vade — production readiness

13. En az 90 tamamlanmış shadow günü biriktir.
14. Günlük expected-race → program snapshot → final prediction → official result → evaluation zincirini reconciliation tablosuyla kapat.
15. Calibration/drift threshold'larını yeterli örnek sayısı koşuluna bağla.
16. CI/CD, secret scanning, exact dependency lock ve Git release süreci kur.
17. Canlı 50k+ arşivde API latency/load benchmark'ı yap.

## 13. Beklenen etkiler

| Düzeltme | Beklenen yön | Tahmini büyüklük / güven |
|---|---|---|
| Current rating'i lagged/pre-race yapmak | Top-1 düşer ama gerçekçi olur | Kontrollü CatBoost'ta `-29.4 puan`; yüksek güven |
| Tam starter coverage | Top-1/ROI düşebilir, karşılaştırma geçerli olur | Yön yüksek güven, büyüklük yeniden build olmadan bilinemez |
| XGB tarih parser düzeltmesi | Artifact/backtest uyumu artar | Büyük; exact etki yeniden eğitim ister |
| OOD yabancı pistleri ayırmak | Yanlış güven azalır | Yüksek güven |
| Empty coverage PASS düzeltmesi | Yanlış sağlık sinyali azalır | Kesin |
| 90 günlük shadow | Canlı calibration/accuracy ölçülebilir | Kesin, skor seviyesi bilinemez |

Bu aşamada doğruluk artışı vaat edilmemelidir. İlk düzeltmeler muhtemelen raporlanan doğruluğu düşürecektir; bu başarısızlık değil, sahte iyimserliğin kaldırılmasıdır.

## 14. İlk yapılması gereken 10 iş

1. Backtest v2 ve ROI v2'yi `NOT CERTIFIED — post-race rating leakage` etiketiyle dondur.
2. Model feature contract'ta current-race `handicap_rating` için fail-closed provenance kuralı tanımla.
3. 2026 için tam starter coverage reconciliation raporu üret.
4. Kazananı veya starter'ları eksik yarışları train/eval'den sessizce seçmek yerine build'i fail ettir.
5. XGBoost production artifact'ını kaldırmadan deprecated işaretle ve doğru tarih parser ile yeniden üret.
6. CatBoost/Logistic artifact'larının eğitim kaynağını manifestle; kanıtlanamıyorsa deprecated yap.
7. Snapshot coverage'ın `0 prediction = PASS` davranışını düzelt.
8. VPS credential rotate + secret scan yap.
9. Güncel VPS timer/result/prediction zincirini en az bir tam yarış gününde acceptance test ile doğrula.
10. Leakage-safe dataset hazır olmadan feature ekleme, SHAP yorumlama veya bahis optimizasyonu yapma.

## 15. Nihai karar

**Production ready: HAYIR.**  
**Canlı bahis için uygun: HAYIR.**  
**Shadow altyapısı değerlendirilmeye değer mi: EVET, fakat veri akışı henüz kanıtlanmadı.**  
**Mevcut `%57.95` Top-1 ve `%125.33` ROI geçerli mi: HAYIR; deprecated/not-certified olmalı.**

Projeyi çöpe atmak gerekmiyor. En değerli parça model skoru değil; immutable snapshot, read-only dashboard ve lifecycle yönündeki altyapı. Doğru sonraki adım yeni feature eklemek değil, program-bazlı tam yarış evreni ve gerçekten pre-race rating ile bilimsel zemini yeniden kurmaktır.

## 16. Kanıt kaynakları

- `pedigreeall_progress.db`
- `build_final_dataset.py`
- `build_asof_features.py`
- `feature_contract.py`
- `run_production_backtest.py`
- `train_xgboost_production.py`
- `shadow_mode.py`
- `shadow_monitor.py`
- `validate_feature_provenance.py`
- `performance_queries.py`
- `bet_simulator_queries.py`
- `deploy/systemd/*`
- `reports/backtest_report_v2.md`
- `reports/model_comparison_v2.md`
- `reports/feature_importance_v2.md`
- `reports/leakage_gate_v2.md`
- `reports/snapshot_coverage.md`
- `reports/vps_deploy_validation_20260629_121151.md` (credential değeri bu rapora taşınmadı)

