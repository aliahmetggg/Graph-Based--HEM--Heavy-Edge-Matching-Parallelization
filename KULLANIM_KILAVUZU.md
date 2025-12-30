# CUDA Parallel Sparse Matrix Scaling - Kullanım Kılavuzu

**Yazarlar:** Ali Ahmet Taşkesen, Ömer Yıldırım  
**Üniversite:** Ankara Yıldırım Beyazıt Üniversitesi

---

## 📋 Gereksinimler

- Windows 10/11
- NVIDIA GPU (RTX serisi önerilir)
- Visual Studio 2022 Community
- CUDA Toolkit 12.6

x64 Native Tools Command Prompt for VS 2022 
terminalini admin olarak çalıştırmamız gerekmektedir
---

## 🔧 Kurulum (Tek Seferlik)

### 1. Visual Studio 2022 Community Yükle

1. https://visualstudio.microsoft.com/downloads/ adresine git
2. **Community** (ücretsiz) indir
3. Kurulumda **"Desktop development with C++"** seç ✅
4. Yükle ve bilgisayarı yeniden başlat

### 2. CUDA Toolkit 12.6 Yükle

1. https://developer.nvidia.com/cuda-12-6-0-download-archive adresine git
2. **Windows → x86_64 → 11 → exe (local)** seç
3. İndir ve yükle
4. Bilgisayarı yeniden başlat

### 3. Kurulumu Doğrula

**x64 Native Tools Command Prompt for VS 2022** aç:
- Windows arama → "x64 Native Tools Command Prompt for VS 2022"

```cmd
nvcc --version
```

Çıktı şöyle olmalı:
```
Cuda compilation tools, release 12.6, V12.6.20
```

---

## 🔨 Derleme

### 1. x64 Native Tools Command Prompt Aç

Windows arama → **"x64 Native Tools Command Prompt for VS 2022"**

### 2. Proje Klasörüne Git

```cmd
cd C:\cuda_test
```
(veya dosyaların olduğu klasör)

### 3. Library Path Ayarla

```cmd
set LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\lib\x64;%LIB%
```

> **Not:** `14.41.34120` klasör adı sisteminizde farklı olabilir. Kontrol etmek için:
> ```cmd
> dir "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
> ```

### 4. Derle

**NN Metodu:**
```cmd
nvcc -O3 -arch=sm_89 -o parallel_nn.exe parallel_nn_v2.cu
```

**DCT Metodu:**
```cmd
nvcc -O3 -arch=sm_89 -lcufft -o parallel_dct.exe parallel_dct.cu
```

**Graph-HEM Metodu (YENİ!):**
```cmd
nvcc -O3 -arch=sm_89 -o parallel_graph_hem.exe parallel_graph_hem.cu
```

> **Not:** DCT için `-lcufft` eklenmeli (cuFFT kütüphanesi kullanıyor)

> **GPU Mimarisi:** RTX 4060/4070/4080/4090 için `sm_89` kullanın.
> Diğer GPU'lar için:
> - RTX 3060/3070/3080: `sm_86`
> - RTX 2060/2070/2080: `sm_75`

Başarılı çıktı:
```
Creating library parallel_nn.lib and object parallel_nn.exp
```

---

## 🚀 Çalıştırma

### Türkçe Karakter Düzeltmesi (Önemli!)

**Geçici Çözüm:** Her terminalde ilk önce şunu çalıştır:
```cmd
chcp 65001
```

**Kalıcı Çözüm:** Bu komutu bir kere çalıştır, terminali kapat-aç:
```cmd
reg add "HKCU\Console" /v CodePage /t REG_DWORD /d 65001 /f
```
Artık her terminal açıldığında otomatik UTF-8 olacak!

### Temel Kullanım

```cmd
parallel_nn.exe -s 5000 -v
parallel_graph_hem.exe -s 5000 -v
```

**Parametreler:**
| Parametre | Açıklama | Örnek |
|-----------|----------|-------|
| `-s <boyut>` | Matris boyutu | `-s 5000` → 5000×5000 |
| `-b <bant>` | Bandwidth (şerit genişliği) | `-b 10` |
| `-r <tekrar>` | Benchmark tekrar sayısı | `-r 10` |
| `-v` | Doğrulama modu (GPU vs CPU) | `-v` |

### Örnek Testler

```cmd
# Küçük test (hızlı)
parallel_nn.exe -s 1000 -v
parallel_graph_hem.exe -s 1000 -v

# Orta test
parallel_nn.exe -s 5000 -v
parallel_graph_hem.exe -s 5000 -v

# Büyük test
parallel_nn.exe -s 10000 -v
parallel_graph_hem.exe -s 10000 -v

# Çok büyük test
parallel_nn.exe -s 20000 -v
parallel_graph_hem.exe -s 20000 -v
```

---

## 📊 Çıktıyı Anlama

```
┌───────────────────┬────────────┬───────────┬───────────┬─────────┬────────┐
│ Operation         │ Target     │ GPU (ms)  │ CPU (ms)  │ Speedup │ Errors │
├───────────────────┼────────────┼───────────┼───────────┼─────────┼────────┤
│ Expand (+1)       │ 5001x5001  │     0.013 │     1.800 │ 142.91x │      0 │
│ Upscale (2x)      │ 10k x 10k  │     0.030 │     1.700 │  56.43x │      0 │
│ Reduce (-1)       │ 4999x4999  │     0.016 │     1.700 │ 107.26x │      0 │
│ Downscale (1/2x)  │ 2500x2500  │     0.031 │     1.700 │  54.15x │      0 │
│ Upscale (4x)      │ 20k x 20k  │     0.016 │     1.700 │ 104.39x │      0 │
└───────────────────┴────────────┴───────────┴───────────┴─────────┴────────┘
```

| Sütun | Açıklama |
|-------|----------|
| **Operation** | Ölçekleme işlemi türü |
| **Target** | Hedef matris boyutu |
| **GPU (ms)** | GPU'da geçen süre (milisaniye) |
| **CPU (ms)** | CPU'da geçen süre (milisaniye) |
| **Speedup** | Hızlanma oranı (CPU/GPU) |
| **Errors** | Hata sayısı (0 olmalı) |

**İşlem Türleri:**
- **Expand (+1):** Boyutu 1 artır (N → N+1)
- **Upscale (2x):** Boyutu 2 katına çıkar (N → 2N)
- **Reduce (-1):** Boyutu 1 azalt (N → N-1)
- **Downscale (1/2x):** Boyutu yarıya indir (N → N/2)
- **Upscale (4x):** Boyutu 4 katına çıkar (N → 4N)

---

## 🧪 Test Matrisi Hakkında

Program **banded (şeritli) sparse matris** oluşturur:

```
Örnek 5×5 banded matris (bandwidth=2):

    [ 1.0  0.9  0.8   0    0  ]
    [ 0.9  1.0  0.9  0.8   0  ]
    [ 0.8  0.9  1.0  0.9  0.8 ]
    [  0   0.8  0.9  1.0  0.9 ]
    [  0    0   0.8  0.9  1.0 ]
```

**Neden banded matris?**
- Mühendislik uygulamalarında yaygın (FEM, ısı transferi)
- SuiteSparse koleksiyonundaki matrislere benzer
- Parametrik olarak boyut ve density ayarlanabilir

---

## 📈 Beklenen Sonuçlar (RTX 4060)

### NN Metodu

| Matris Boyutu | NNZ | Ortalama Speedup |
|---------------|-----|------------------|
| 5K × 5K | ~55K | **25-30x** |
| 10K × 10K | ~110K | **25-30x** |
| 20K × 20K | ~220K | **25-30x** |

### Graph-HEM Metodu (YENİ!)

| Matris Boyutu | NNZ | Ortalama Speedup |
|---------------|-----|------------------|
| 500 × 500 | ~5.5K | **4-5x** |
| 2K × 2K | ~22K | **28x** |
| 5K × 5K | ~55K | **67x** |
| 10K × 10K | ~110K | **113x** |
| 20K × 20K | ~220K | **171x** |

> **Not:** Graph-HEM büyük matrislerde çok daha iyi ölçeklenir!

---

## 🔬 NN vs DCT vs Graph-HEM Karşılaştırma Benchmark

Üç exe derlendikten sonra karşılaştırma yapmak için:

**Manuel Test:**
```cmd
chcp 65001
parallel_nn.exe -s 5000 -v
parallel_dct.exe -s 5000 -v
parallel_graph_hem.exe -s 5000 -v
```

**Otomatik Benchmark (run_tests.bat):**
```cmd
run_tests.bat
```

### NN vs DCT vs Graph-HEM Farkları:

| Özellik | NN Metodu | DCT Metodu | Graph-HEM Metodu |
|---------|-----------|------------|------------------|
| **Hız** | Hızlı (~25x) | Orta (~19x) | **Çok Hızlı (~171x)** |
| **Bellek** | Düşük O(nnz) | Yüksek O(n²) | Orta O(nnz + E) |
| **NNZ Korunumu** | Mükemmel | Threshold'a bağlı | ~63% (clustering) |
| **Kullanım Alanı** | Genel amaç | Yüksek kalite upscale | Büyük matrisler |
| **Kütüphane** | Sadece CUDA | CUDA + cuFFT | Sadece CUDA |
| **Algoritma** | Koordinat ölçekleme | Frekans domain | Graf kümeleme |

### Speedup Karşılaştırması (20K×20K matris):

| İşlem | NN | DCT | Graph-HEM |
|-------|-----|-----|-----------|
| Expand (+1) | 26.6x | 18.4x | **171x** |
| Upscale (2x) | 25.4x | 20.1x | **166x** |
| Reduce (-1) | 23.5x | 17.2x | **177x** |
| Downscale (1/2x) | 24.8x | 19.6x | **170x** |
| **Ortalama** | 25.1x | 18.8x | **171x** |

> **Sonuç:** Graph-HEM, NN ve DCT'den **~7 kat** daha hızlı!

---

## 🧠 Graph-HEM Nasıl Çalışır?

Graph-HEM (Graph-based Heavy-Edge Matching) metodu, sparse matrisi bir graf olarak ele alır:

### Algoritma Adımları:

```
1. Her nonzero element → Graf düğümü
2. 4-bağlantılılık (komşuluk) → Graf kenarları
3. Heavy-Edge Matching → En ağır komşuyla eşleş
4. Kümeleme → Eşleşen düğümleri birleştir
5. Ölçekleme → Küme merkezlerini yeni boyuta taşı
```

### 4-Bağlantılılık (Von Neumann Komşuluğu):

```
        (i-1, j)
           ↑
(i, j-1) ← ● → (i, j+1)
           ↓
        (i+1, j)
```

### CUDA Kernel Pipeline (8 kernel):

| # | Kernel | Görev |
|---|--------|-------|
| 1 | `build_nodes` | Düğüm koordinatlarını çıkar |
| 2 | `create_pos_map` | Pozisyon-düğüm haritası |
| 3 | `count_neighbors` | Komşu sayısını say |
| 4 | `build_adjacency` | Kenar listesi oluştur |
| 5 | `find_best_match` | En iyi eşleşmeyi bul |
| 6 | `perform_matching` | Paralel eşleştirme (atomicCAS) |
| 7 | `assign_clusters` | Küme ID'leri ata |
| 8 | `scale_clusters` | Merkezleri ölçekle |

### Neden Bu Kadar Hızlı?

1. **Daha İyi Paralelizm:** Her düğüm bağımsız işlenebilir
2. **Azaltılmış Çıktı:** ~37% sıkıştırma (220K → 138K)
3. **Cache-Friendly:** 4-bağlantılılık yerel erişim sağlar
4. **Verimli Atomics:** Modern GPU'larda atomicCAS çok hızlı

---

## ❗ Sık Karşılaşılan Hatalar

### Hata: "nvcc not found"
**Çözüm:** CUDA Toolkit yüklü değil veya PATH'e eklenmemiş. CUDA'yı yeniden yükle.

### Hata: "cl.exe not found"
**Çözüm:** Normal CMD yerine **x64 Native Tools Command Prompt for VS 2022** kullan.

### Hata: "LIBCMT.lib not found"
**Çözüm:** `set LIB=...` komutunu çalıştır (yukarıdaki Derleme bölümüne bak).

### Hata: "cudafe++ ACCESS_VIOLATION"
**Çözüm:** CUDA 12.6 yükle (13.x sürümleri buglu olabilir).

### Türkçe karakterler bozuk görünüyor
**Çözüm:** `chcp 65001` komutunu çalıştır.

---

## 🔨 EXE Nasıl Oluşuyor? (Derleme Süreci)

### Derleme Komutu Açıklaması:
```cmd
nvcc -O3 -arch=sm_89 -o parallel_nn.exe parallel_nn_v2.cu
```

| Parça | Açıklama |
|-------|----------|
| `nvcc` | NVIDIA CUDA Compiler |
| `-O3` | Optimizasyon seviyesi 3 (en hızlı kod) |
| `-arch=sm_89` | GPU mimarisi (RTX 4060 = sm_89) |
| `-o parallel_nn.exe` | Çıktı dosyası adı |
| `parallel_nn_v2.cu` | Kaynak kod (.cu = CUDA C++) |

### Derleme Aşamaları:

```
parallel_nn_v2.cu  (CUDA kaynak kodu)
        │
        ▼
   ┌─────────┐
   │  nvcc   │  ← NVIDIA derleyici
   └────┬────┘
        │ (2 aşama)
        ▼
┌─────────────────────────────────────────┐
│ 1. GPU Kodu → PTX → cubin (GPU binary)  │
│ 2. CPU Kodu → cl.exe (VS) → .obj        │
└─────────────────────────────────────────┘
        │
        ▼
   ┌─────────┐
   │ LINKER  │  ← Hepsini birleştirir
   └────┬────┘
        │
        ▼
  parallel_nn.exe
```

### Linker Nedir?

**Linker**, derlenmiş kod parçalarını birleştiren programdır.

Basit analoji - araba montajı gibi düşün:
- **Motor** → GPU kodu (kernel'lar)
- **Karoseri** → CPU kodu (main fonksiyon)
- **Tekerlekler** → CUDA kütüphaneleri (cudaMalloc, cudaFree)
- **LINKER** → Montaj hattı (hepsini birleştirip çalışan araba yapar)

Birleştirilen parçalar:
| Parça | İçerik |
|-------|--------|
| GPU Kodu (.cubin) | kernel fonksiyonları |
| CPU Kodu (.obj) | main, yardımcı fonksiyonlar |
| CUDA Runtime (.lib) | cudaMalloc, cudaFree, cudaMemcpy |
| Sistem kütüphaneleri | LIBCMT.lib, vs. |

---

## 📁 Dosya Yapısı

```
parallel_sparse_matrix_scaling/
├── parallel_nn_v2.cu          # NN metodu (CUDA)
├── parallel_dct.cu            # DCT metodu (cuFFT)
├── parallel_graph_hem.cu      # Graph-HEM metodu (YENİ!)
├── parallel_nn_python.py      # Python versiyonu
├── dct_scaling.py             # Python DCT
├── README.md                  # İngilizce dokümantasyon
├── KULLANIM_KILAVUZU.md       # Bu dosya
├── run_tests.bat              # Windows test scripti
└── paper_updated.pdf          # Akademik makale
```

---

## 📚 Akademik Referans

Bu proje hakkında detaylı bilgi için akademik makaleyi okuyabilirsiniz:

**"Parallel Sparse Matrix Scaling Using CUDA: Accelerating Nearest Neighbor, DCT, and Graph-Based Methods"**

Makale şu konuları içerir:
- NN, DCT ve Graph-HEM algoritmalarının teorik temelleri
- CUDA implementasyon detayları
- Performans karşılaştırmaları ve analiz
- Karypis & Kumar'ın Heavy-Edge Matching algoritması

---

## 📞 İletişim

- **Ali Ahmet Taşkesen:** aliahmetaskesen@gmail.com
- **Ömer Yıldırım:** flashomer@gmail.com

---

*Ankara Yıldırım Beyazıt Üniversitesi - Paralel Programlama Dersi - 2025*
