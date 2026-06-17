Folder ini harus berisi source repo HuggingFace asli:
https://huggingface.co/spaces/mothieram/face-anti-spoofing

Yang wajib ada di folder ini sebelum FastAPI dijalankan:
- IADG.py
- SASF.py
- infer_cdcnpp.py
- CDCNPP.py
- liveness_temporal.py
- detector.py
- models.py
- tsn_predict.py
- folder src/
- folder weights/
- folder finetuned_weights/ (sudah berisi weight fine-tuned dari model baru.zip)

Cara mudah:
1) Download/clone repo HF ke folder sementara.
2) Copy semua isi repo tersebut ke folder ini.
3) Jangan hapus folder finetuned_weights yang sudah ada; jika tertimpa, copy ulang weight fine-tuned.
