---
title: Face Recognition Microservice HRIS
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# Face Recognition Microservice — HRIS (Hotel Labersa Toba)
Microservice **FastAPI** untuk:
- Ekstraksi embedding wajah (FaceNet)
- Verifikasi selfie vs embedding acuan
- Anti-spoofing (ensemble dari folder `face-anti-spoofing_hf`) + validasi tambahan (screen spoof, aksesoris)

Service ini **stateless** (tanpa database) dan dipanggil oleh backend internal (misalnya Golang) memakai header `X-API-Key`.

- Panduan instalasi & menjalankan: [Instalasi.md](file:///d:/Semester%206/PA/HRIS-FastAPI/Instalasi.md)
- Catatan integrasi anti-spoofing HF: [INTEGRASI_HF_ANTISPOOF_FASTAPI.md](file:///d:/Semester%206/PA/HRIS-FastAPI/INTEGRASI_HF_ANTISPOOF_FASTAPI.md)

## Model & Folder Penting
- `models/facenet_labersa_cpu.pt`: checkpoint FaceNet/classifier (opsional; jika tidak ada, tetap memakai backbone pretrained VGGFace2)
- `face-anti-spoofing_hf/`: source + weights anti-spoofing (ensemble)
- `.env.example`: contoh konfigurasi runtime

## Endpoint
Semua endpoint **kecuali** `/health` membutuhkan header: `X-API-Key: <FACE_API_KEY>`.

| Method | Path | Deskripsi |
|---|---|---|
| GET | `/health` | Status service, config, dan status load model |
| POST | `/face/extract` | Ekstrak embedding dari foto wajah (registrasi) |
| POST | `/face/verify` | Verifikasi selfie vs embedding acuan (absen) |
| GET | `/face/antispoof/config` | Lihat threshold & bobot ensemble (internal) |
| PUT | `/face/antispoof/config` | Ubah threshold & bobot ensemble (internal) |
| POST | `/face/antispoof/reload` | Reload model ensemble (internal) |
| POST | `/face/antispoof/predict` | Tes anti-spoofing tanpa verifikasi embedding |

## Konfigurasi (.env)
Konfigurasi utama ada di `.env`. Minimal yang perlu diset:
- `FACE_API_KEY`: API key internal untuk akses endpoint
- `PORT`: port yang digunakan saat menjalankan server (contoh: `8001`)
- `DEVICE`: `cpu` atau `cuda` (jika tersedia)
- `HF_REPO_DIR`: path ke folder `face-anti-spoofing_hf`
- `ANTI_SPOOFING_ENABLED`: `1` untuk aktif, `0` untuk nonaktif

## Jalankan Cepat (Tanpa Docker)
Ikuti langkah lengkap di [Instalasi.md](file:///d:/Semester%206/PA/HRIS-FastAPI/Instalasi.md). Ringkasnya:

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8001
```

## Jalankan Dengan Docker
Dockerfile default menjalankan uvicorn di port `7860` (sesuai konfigurasi deployment). Contoh menjalankan service di host port `8001`:

```bash
docker build -t hris-face-service .
docker run --rm -p 8001:7860 --env-file .env hris-face-service
```

## Integrasi Backend (Golang)
Untuk registrasi wajah gunakan `/face/extract`, dan untuk absensi gunakan `/face/verify`.

Repository ini menyediakan contoh client: [face_client.go](file:///d:/Semester%206/PA/HRIS-FastAPI/face_client.go) (bagian `/face/extract` bisa langsung dipakai). Jika backend Anda masih memakai endpoint `/geo/*` atau `/attendance/*`, sesuaikan karena endpoint tersebut tidak tersedia pada versi service ini.

## Keamanan
- Jangan expose service ini langsung ke publik; batasi akses hanya dari backend internal.
- Gunakan `FACE_API_KEY` yang kuat dan simpan sebagai secret (jangan commit `.env`).
