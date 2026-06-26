# Panduan Instalasi & Menjalankan Project

Dokumen ini menjelaskan langkah-langkah menjalankan **Face Recognition Microservice (FastAPI)** di lokal (Windows/Linux/macOS) dan via Docker.

## Prasyarat

- Python 3.10+ (disarankan 3.10/3.11)
- Git
- Opsional: Docker Desktop / Docker Engine (jika ingin run via container)

## 1) Menjalankan Secara Lokal (Tanpa Docker)

### A. Windows (PowerShell)

```powershell
cd "D:\Semester 6\PA\HRIS-FastAPI"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
```

Edit file `.env` sesuai kebutuhan (minimal `FACE_API_KEY`, `PORT`, dan `HF_REPO_DIR`).

Jalankan server:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8001
```

Jika ingin mengikuti value `PORT` dari `.env`, jalankan:

```powershell
python main.py
```

### B. Linux / macOS (bash/zsh)

```bash
cd /path/ke/HRIS-FastAPI

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Jalankan server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

Atau mengikuti `PORT` dari `.env`:

```bash
python main.py
```

## 2) Konfigurasi Environment (.env)

File contoh ada di `.env.example`. Variabel yang paling sering dipakai:

- `FACE_API_KEY`: API key internal (wajib untuk semua endpoint kecuali `/health`)
- `PORT`: port server saat menjalankan `python main.py` (contoh: `8001`)
- `DEVICE`: `cpu` atau `cuda` (jika GPU tersedia)
- `MODEL_PATH`: default `models/facenet_labersa_cpu.pt`
- `ANTI_SPOOFING_ENABLED`: `1` aktif, `0` nonaktif
- `HF_REPO_DIR`: folder `face-anti-spoofing_hf` (default sudah sesuai jika folder ada di root project)
- `HF_ANTISPOOF_CONFIG_PATH`: default `api_settings.json`

## 3) Cek Model & Folder

Pastikan file/folder ini ada:

- `models/facenet_labersa_cpu.pt` (opsional; jika tidak ada service tetap jalan memakai backbone pretrained)
- `face-anti-spoofing_hf/` (dibutuhkan jika `ANTI_SPOOFING_ENABLED=1`)

## 4) Verifikasi Service (Health Check)

```bash
curl http://127.0.0.1:8001/health
```

Jika sudah berjalan, response berisi `status: "ok"` dan `model_loaded: true/false` (biasanya `true` setelah startup selesai).

## 5) Contoh Request Endpoint

### A. Extract Embedding (Registrasi)

```bash
curl -X POST "http://127.0.0.1:8001/face/extract" \
  -H "X-API-Key: labersa-internal-api-key-2026" \
  -F "employee_id=EMP001" \
  -F "photo=@selfie.jpg"
```

### B. Verify Face (Absensi)

Endpoint `/face/verify` membutuhkan:

- `photo`: file selfie terbaru
- `data`: JSON string berisi `employee_id`, `stored_embedding` (512 float), dan opsional `threshold`
- `liveness`: string `"true"` (jika `LIVENESS_ENABLED=1`)

```bash
curl -X POST "http://127.0.0.1:8001/face/verify" \
  -H "X-API-Key: labersa-internal-api-key-2026" \
  -F 'data={"employee_id":"EMP001","stored_embedding":[0.0,0.0,0.0],"threshold":0.75}' \
  -F "liveness=true" \
  -F "photo=@selfie.jpg"
```

Catatan: `stored_embedding` wajib 512 dimensi. Contoh di atas hanya placeholder.

## 6) Menjalankan Dengan Docker

Dockerfile default menjalankan uvicorn di port `7860` (berguna untuk deployment). Contoh menjalankan di host port `8001`:

```bash
docker build -t hris-face-service .
docker run --rm -p 8001:7860 --env-file .env hris-face-service
```

Health check:

```bash
curl http://127.0.0.1:8001/health
```

## 7) Troubleshooting Singkat

- Torch install lama: pastikan koneksi internet stabil; requirements sudah memakai CPU wheels via `--extra-index-url`.
- Anti-spoofing error/path: pastikan `HF_REPO_DIR` menunjuk ke folder `face-anti-spoofing_hf` yang berisi `src/` dan `weights/`.
- Port bentrok: ubah `PORT` di `.env`, atau jalankan uvicorn dengan `--port` yang lain.

