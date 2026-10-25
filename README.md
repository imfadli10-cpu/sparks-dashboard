# Dasbor SPARKS

Dasbor Streamlit untuk **SPARKS — Stochastic Peatland Adaptive Risk & Kalman System**, peringatan dini
kekeringan lahan gambut **seluruh Indonesia per kabupaten/kota (514; 119 bergambut) atau per provinsi (38), dengan
pilihan pulau**, dengan data **2015 sampai ±4 hari lalu**. Semua isian ada di baris filter di atas: Pulau · Dari/Sampai
tanggal · Ambil data · Kondisi per tanggal · pengaturan (Project ID).

| Tab | Isi | Untuk |
|---|---|---|
| Situasi & Peringatan | ringkasan, angka utama, peta (peringatan kebakaran XGBoost atau peluang gambut kering), daftar kab/kota yang perlu tindakan, detail kab/kota | pemerintah |
| Perkembangan | luas gambut kering dan titik api per minggu | pemerintah |
| Rencana Patroli | biaya patroli per status (Awas/Siaga/Waspada), anggaran minggu ini, kab/kota yang terjangkau anggaran, urutan prioritas patroli (CSV) | pemerintah |
| Tentang Model | cara kerja dan batasan; akurasi model XGBoost (ROC, ketepatan peringatan); prediksi vs kenyataan; faktor pendorong (SHAP) | analis / juri |

Baris atas: **Wilayah · Kondisi minggu · Prediksi untuk (1–4 minggu) · Data** (rentang tanggal, ambil data dari satelit, Project ID).

## Menjalankan

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Isi **Project ID Google Earth Engine** di tombol pengaturan (kanan baris filter), lalu pilih **rentang tanggal**
(bebas, sejak Mei 2015) dan **Kondisi per tanggal** (Senin mana yang ditampilkan).

- **Mengubah tanggal langsung mengubah tampilan** dari kumpulan data yang sudah pernah diambil (tanpa unduh).
- Bila sebagian tanggal di rentang itu belum pernah diambil, baris filter menyebut tanggal mana saja; **Ambil data**
  lalu hanya mengunduh tanggal itu dari Earth Engine (mis. sudah ada 23–30 Des, minta 1–30 Des → yang diunduh 1–22 Des),
  kemudian mengolah ulang SEMUA data tersimpan menjadi satu kumpulan bersambung.
- Unduhan mentah disimpan di `hasil/mentah_smap/` (nama berkas = jenis_tanggalawal_tanggalakhir_wilayah) dan tidak
  pernah diunduh ulang. Satu tahun data seluruh Indonesia memakai ±25 EECU-jam kuota Earth Engine (tier Community:
  150 EECU-jam per bulan per project).

```bash
python app.py proses --mulai 2025-01-01 --selesai 2026-10-13         # data dasbor tanpa membuka dasbor
python app.py esai                                                     # analisis Bab III esai -> hasil/
```

## Sumber data (semua lewat Google Earth Engine)

| Data | Dataset | Pembaruan |
|---|---|---|
| Muka air tanah gambut, hujan, evapotranspirasi, suhu | NASA SMAP L4 v008 (`NASA/SMAP/SPL4SMGP/008`), 9 km, tiap 3 jam | terlambat ±3–5 hari |
| Titik api | NASA FIRMS (`FIRMS`, MODIS ±1 km), harian; hanya keyakinan ≥ 30 (nominal dan tinggi) | terlambat ±1–2 hari |
| Batas kabupaten/kota | FAO GAUL 2025 level 2 | tetap |

**Variabel model:** XGBoost memakai 9 variabel numerik: peluang kering, muka air, perubahan muka air 1 dan 2 minggu, hujan 30 hari, Drought Code, titik api 1 dan 2 minggu lalu, dan minggu dalam setahun.

## Cara menghitung (BAGIAN 4 dan 5 di `app.py`)

1. **Muka air gambut** harian h (m, negatif = di bawah permukaan) untuk setiap piksel gambut SMAP 9 km
   (pita `depth_to_water_table_from_surface_in_peat`, hanya terisi di piksel gambut PEATCLSM).
2. **Cuaca per kabupaten/kota**: hujan dan evapotranspirasi (kg m⁻² s⁻¹ × 86.400 = mm/hari) dan suhu maksimum
   harian (K − 273,15). Tiap piksel memakai cuaca kabupatennya; kab/kota yang terlalu kecil untuk piksel 9 km memakai
   rata-rata provinsinya.
3. **Drought Code** (Van Wagner 1987; faktor panjang hari 1,4 untuk lintang ±10°) dihitung **bersambung** sepanjang
   deret, tidak diulang tiap 1 Januari. Suhu maksimum harian dipakai sebagai pengganti suhu pukul 12 siang [A].
4. **Filter Kalman** 1-D per piksel: h_t = h_{t−1} + a·P_t − b·ET_t − c·h_{t−1} + η_t, observasi z_t = h_t + ε_t
   (muka air SMAP). Parameter (a, b, c, Q, R) diestimasi dengan MLE (dekomposisi galat prediksi) per provinsi pada
   blok tanggal bersambung terpanjang di data tersimpan; tiap blok (dipisah bila ada lompatan > 14 hari) difilter
   terpisah. Catatan: MLE menaksir R ≈ 0 karena keluaran SMAP L4 sendiri mulus (tanpa derau acak harian), jadi
   filter praktis mengikuti SMAP.
5. **Peluang kering** p = Φ((−0,40 − ĥ)/σ), σ² = ragam filter + σ_SMAP², dengan σ_SMAP = 0,10 m [A] = galat
   sistematik muka air SMAP L4 terhadap lapangan. Galat ini menetap dari hari ke hari, sehingga tidak bisa dihaluskan
   filter. Batas −0,40 m sesuai PP 57/2016. Nilai kabupaten = rata-rata piksel gambutnya (= perkiraan bagian lahan
   gambut yang kering); nilai provinsi dan rata-rata wilayah ditimbang **luas gambut**.
6. **Label "kebakaran meluas"**: ≥ 1 piksel api FIRMS (keyakinan ≥ 30) **di dalam piksel gambut** per 10.000 ha
   **lahan gambut** kabupaten dalam k minggu berikutnya (k = 1–4).
7. **Level kekeringan**: ambang dari minimisasi c_FN·FN + c_FP·FP (rasio 2 : 1 untuk Siaga; 4 : 1 Waspada; 1 : 1
   Awas) terhadap label k = 2 minggu.
8. **Model XGBoost** (300 pohon, kedalaman 3, laju belajar 0,05) dengan **batasan monoton** sesuai ilmu gambut (risiko tidak turun bila peluang kering, Drought Code, atau titik api naik; tidak naik bila muka air atau hujan naik; waktu dalam tahun tidak dibatasi). Fitur pada
   minggu t: peluang kering dan muka air (Kalman), perubahan muka air 1 dan 2 minggu, hujan 30 hari, Drought Code,
   titik api gambut per 10.000 ha gambut pada minggu t−1 dan t−2, dan minggu dalam setahun. Model akhir dilatih
   pada kabupaten-minggu ≤ t − (k+1) minggu (label k minggu ke depannya sudah lengkap teramati pada Senin t).
9. **Evaluasi maju-waktu** (rolling origin): minggu latih dibagi 6 blok berurutan; tiap blok (kecuali yang pertama)
   ditebak model yang hanya dilatih pada minggu ≤ awal blok − (k+1) minggu. Ambang level, presisi, recall, F1, AUC,
   kurva ROC, dan confusion matrix dari prediksi ini. SK 95% AUC dari bootstrap berklaster per minggu
   (300 kali). Penjelasan model dengan SHAP (TreeSHAP XGBoost, satuan log-odds).

### Asumsi [A] yang perlu disebut

- σ_SMAP = 0,10 m (galat sistematik SMAP L4 terhadap muka air lapangan; perlu dirujuk ke evaluasi PEATCLSM).
- Rasio biaya melewatkan kebakaran : peringatan palsu = 2 : 1; ambang "kebakaran meluas" 1 piksel api / 10.000 ha gambut.
- Bila belum ada riwayat titik api, ambang level 25% / 50% / 75%.
- Suhu maksimum harian sebagai pengganti suhu siang pada Drought Code.
- Rekomendasi tindakan per status bersifat umum (Awas: patroli harian, pembasahan sekat kanal, pemadaman dini; Siaga: siapkan regu dan peralatan; Waspada: pantau titik api) dan perlu disesuaikan dengan SOP daerah.

## Hubungan dengan esai (blueprint)

Mesin dasbor memakai SMAP L4, sehingga peluang kering dinyatakan langsung dalam meter (Pr(muka air < −0,4 m)), bukan
indeks relatif. Analisis Bab III blueprint (kotak pilot Sentinel-1 100 m di KHG Kahayan–Sebangau, latih 2018 / uji 2019,
Juni–Oktober: MLE, RMSE vs persistensi, AUC/DeLong vs Drought Code, tangkapan titik panas) tetap ada di BAGIAN 3 dan
dijalankan dengan `python app.py esai`.

## Batasan

- Resolusi SMAP 9 km: satu piksel mewakili ±8.100 ha dan seluruhnya dianggap gambut; keragaman di dalam satu
  kabupaten disederhanakan.
- Muka air SMAP L4 adalah keluaran model (PEATCLSM) yang mengasimilasi pengamatan satelit, bukan pengukuran
  lapangan; belum dikalibrasi ke SIPALAGA.
- Titik api FIRMS di Earth Engine berasal dari MODIS (±1 km), lebih kasar daripada VIIRS 375 m.
- Skor risiko kebakaran adalah keluaran model yang belum dikalibrasi sebagai peluang.
- "Minggu dalam setahun" termasuk fitur terkuat; dengan data < 3 tahun model sebagian belajar pola musim/kalender.

## Uji

```bash
python -m pytest test_app.py -q
```
