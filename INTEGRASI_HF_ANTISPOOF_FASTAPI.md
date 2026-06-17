# Integrasi FastAPI Lama dengan Anti-Spoofing HF Ensemble

Paket ini sudah saya patch agar `main.py` tidak lagi memakai `models/antispoof_model_improved.pt` ResNet18 lama. Anti-spoofing sekarang dipanggil melalui `hf_antispoof_ensemble.py`, yang membungkus ensemble:

- SASF / MiniFASNet
- FLRGB / ONNX
- ICM2O / PyTorch
- IOM2C / PyTorch
- CDCN++ / PyTorch, opsional jika weight tersedia

## 1. File yang ditambahkan

Di folder API:

- `hf_antispoof_ensemble.py`
- `api_inference.py`
- `api_config.py`
- `api_settings.json`
- `face-anti-spoofing_hf/finetuned_weights/*.pth`

## 2. Syarat penting

Folder `face-anti-spoofing_hf` harus berisi source repo HuggingFace asli:

- `IADG.py`
- `SASF.py`
- `infer_cdcnpp.py`
- `CDCNPP.py`
- `liveness_temporal.py`
- `src/`
- `weights/`

Di paket ini, saya hanya memasukkan weight fine-tuned dari `model baru.zip`. Source repo dan base weights tetap harus Anda copy/download dari repo HuggingFace asli.

## 3. Struktur folder yang disarankan

```text
api_model/
  main.py
  hf_antispoof_ensemble.py
  api_inference.py
  api_config.py
  api_settings.json
  models/
    facenet_labersa_cpu.pt
    label_encoder.json
  face-anti-spoofing_hf/
    IADG.py
    SASF.py
    infer_cdcnpp.py
    CDCNPP.py
    liveness_temporal.py
    src/
    weights/
      ICM2O.pth.tar
      IOM2C.pth.tar
      modelrgb.onnx
      yolov8n-face.onnx
      ...
    finetuned_weights/
      2.7_80x80_MiniFASNetV2_finetuned.pth
      4_0_0_80x80_MiniFASNetV1SE_finetuned.pth
      ICM2O_finetuned.pth
      IOM2C_finetuned.pth
      cdcnpp.pth
```

## 4. Jalankan di Windows PowerShell

```powershell
cd D:\path\ke\api_model
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

$env:FACE_API_KEY="labersa-internal-api-key-2026"
$env:ANTI_SPOOFING_ENABLED="1"
$env:HF_REPO_DIR="D:\Dataset_Spoof\face-anti-spoofing_hf"
# atau jika repo HF ada di dalam project:
# $env:HF_REPO_DIR="face-anti-spoofing_hf"

uvicorn main:app --host 0.0.0.0 --port 8001
```

## 5. Endpoint baru untuk threshold dan weight

### Lihat konfigurasi

```bash
curl -H "X-API-Key: labersa-internal-api-key-2026" \
  http://127.0.0.1:8001/face/antispoof/config
```

### Ubah threshold dan weight tanpa UI

```bash
curl -X PUT "http://127.0.0.1:8001/face/antispoof/config" \
  -H "X-API-Key: labersa-internal-api-key-2026" \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": {
      "sasf": 0.70,
      "flrgb": 0.45,
      "icm2o": 0.564862,
      "iom2c": 0.218523,
      "cdcn": 0.53
    },
    "weights": {
      "sasf": 0.20,
      "flrgb": 0.20,
      "icm2o": 0.20,
      "iom2c": 0.20,
      "cdcn": 0.20
    }
  }'
```

### Reload model

```bash
curl -X POST -H "X-API-Key: labersa-internal-api-key-2026" \
  http://127.0.0.1:8001/face/antispoof/reload
```

### Tes anti-spoof saja

```bash
curl -X POST "http://127.0.0.1:8001/face/antispoof/predict" \
  -H "X-API-Key: labersa-internal-api-key-2026" \
  -F "photo=@selfie.jpg"
```

## 6. Endpoint lama tetap sama

- `POST /face/extract`
- `POST /face/verify`
- `GET /health`

Pada `/face/extract` dan `/face/verify`, alurnya sekarang menjadi:

1. Decode gambar ke RGB
2. MTCNN validasi jumlah wajah
3. Cek aksesoris
4. Screen spoof heuristic lama
5. HF anti-spoof ensemble
6. FaceNet embedding / cosine similarity

## 7. Catatan threshold

Threshold pada model HF adalah threshold untuk `spoof_prob`, artinya:

```text
spoof jika spoof_prob >= threshold
```

Wrapper mengubah hasilnya menjadi:

```text
real_score = 1 - spoof_score
```

Agar pipeline lama tetap cocok, response `/face/verify` tetap mengembalikan:

- `real_score`
- `spoof_score`
- `anti_spoof_threshold`
- `final_score`
