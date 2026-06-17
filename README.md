---
title: Face Recognition Microservice HRIS
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# Face Recognition Microservice — Hotel Labersa Toba
Model dan stateless microservice khusus Face Recognition & Liveness (Anti-Spoofing) & GPS Geofencing.
Dipanggil oleh **Golang Backend** — tidak ada database, tidak ada auth sendiri.
## Model yang Tersedia di Repository ini:
1. `facenet_labersa_cpu.pt`: Model ekstraksi wajah (FaceNet) yang dioptimasi untuk CPU.
2. `antispoof_model_improved.pt`: Model liveness detection untuk mencegah absen menggunakan foto palsu/layar HP.
---
---

## Arsitektur

```
┌──────────────────────────────────────────────────────────────┐
│               Flutter App / React Web                        │
│                                                              │
│  1. Login → ke Golang                                        │
│  2. Kirim foto + GPS → ke Golang                             │
│  3. Terima hasil absensi ← dari Golang                       │
└─────────────────────┬────────────────────────────────────────┘
                      │ HTTPS (public)
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                   GOLANG BACKEND                             │
│                                                              │
│  - Login & JWT (/auth/login)                                 │
│  - Register pegawai (/employees)                             │
│  - Simpan embedding di DB Golang                             │
│  - Catat absensi ke DB Golang                                │
│  - Laporan & dashboard                                       │
│                                                              │
│  Saat registrasi wajah:                                      │
│    foto → [POST /face/extract] → dapat embedding             │
│    simpan embedding di DB Golang                             │
│                                                              │
│  Saat check-in/out:                                          │
│    foto + embedding_dari_DB + GPS                            │
│    → [POST /attendance/process]                              │
│    → dapat approved/rejected                                 │
│    → catat ke DB Golang                                      │
└─────────────────────┬────────────────────────────────────────┘
                      │ HTTP Internal (X-API-Key)
                      │ server-to-server only
                      ▼
┌──────────────────────────────────────────────────────────────┐
│              FASTAPI MICROSERVICE (port 8001)                │
│                                                              │
│  POST /face/extract        ← ekstrak embedding dari foto     │
│  POST /face/verify         ← cocokkan foto vs embedding      │
│  POST /geo/validate        ← cek GPS dalam radius            │
│  POST /attendance/process  ← pipeline GPS + face sekaligus  │
│  GET  /health              ← status service                  │
│                                                              │
│  ✔ Stateless — tidak ada DB                                  │
│  ✔ Dilindungi X-API-Key                                      │
│  ✔ Embedding dikembalikan ke Golang                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Alur Detail

### Registrasi Wajah (sekali per pegawai)

```
Client → Golang: POST /employees/{id}/register-face  [foto]
Golang → FastAPI: POST /face/extract  [foto + employee_id]  (X-API-Key)
FastAPI → Golang: { embedding: [512 float...] }
Golang → DB: UPDATE employees SET face_embedding = [...] WHERE id = ?
Golang → Client: { success: true }
```

### Check-in / Check-out (setiap hari)

```
Client → Golang: POST /attendance/checkin  [foto + lat + lng]  (JWT)
  │
  ├─ Golang: ambil employee.face_embedding dari DB
  ├─ Golang → FastAPI: POST /attendance/process  [foto + embedding + lat + lng]
  │                                               (X-API-Key)
  │
  ├─ FastAPI: cek GPS → jika gagal, return rejected_gps
  ├─ FastAPI: bandingkan wajah → similarity score
  ├─ FastAPI → Golang: { decision: "approved", approved: true, geo: {...}, face: {...} }
  │
  ├─ Golang: if approved → INSERT attendance ke DB
  └─ Golang → Client: { success: true, checkin_time: "08:05:30", status: "present" }
```

---

## Setup

```bash
# Install
pip install -r requirements.txt

# Konfigurasi
cp .env.example .env
# Edit .env — sesuaikan koordinat kantor & path model

# Jalankan
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
```

---

## Endpoints

| Method | Path | Fungsi |
|--------|------|--------|
| `GET` | `/health` | Status service & model |
| `POST` | `/face/extract` | Ekstrak embedding dari foto (registrasi) |
| `POST` | `/face/verify` | Cocokkan foto vs embedding (opsional) |
| `POST` | `/geo/validate` | Validasi GPS saja |
| `POST` | `/attendance/process` | **Pipeline utama**: GPS + Face sekaligus |

Semua endpoint kecuali `/health` butuh header: `X-API-Key: <key>`

---

## Integrasi Golang

Copy file `face_client.go` ke package service Golang Anda.

```go
// Inisialisasi (sekali, di startup)
faceClient := faceservice.NewFaceClient()

// Registrasi wajah
embedding, err := faceClient.ExtractEmbedding(employeeID, photoBytes, "photo.jpg")
// → simpan embedding ke DB Golang

// Absensi
result, err := faceClient.ProcessAttendance(faceservice.ProcessAttendanceRequest{
    EmployeeID:      employee.ID,
    StoredEmbedding: employee.FaceEmbedding,  // dari DB Golang
    Latitude:        req.Latitude,
    Longitude:       req.Longitude,
    RecordType:      "checkin",
}, photoBytes, "selfie.jpg")

if result.Approved {
    // catat absensi ke DB Golang
}
```

---

## Keamanan

- FastAPI **tidak dapat diakses langsung dari client** (hanya Golang)
- Lindungi dengan firewall: port 8001 hanya bisa diakses dari IP server Golang
- Ganti `FACE_API_KEY` di `.env` dengan string acak yang kuat
- Golang menyertakan header `X-API-Key` di setiap request ke FastAPI
