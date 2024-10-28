# Trash Classification
Repositori ini memuat eksperimen klasifikasi gambar dari [Hugging Face Dataset](https://huggingface.co/datasets/garythung/trashnet). Model yang digunakan adalah model pre-trained ResNet18.

## Project Structures
Proyek ini terdiri dari 3 file utama:
1. `resize_image.ipynb`: Notebook yang berisi kode untuk mengambil data dari huggingface kemudian ukuran gambar diubah menjadi (224, 224)
2. `modeling.ipynb`: Notebook yang berisi eksplorasi gambar, pra-pemrosesan gambar, augmentasi, pembuatan model, hingga evaluasi gambar. Di dalam file ini juga berisi kode untuk push model ke huggingface dan wandb serta melakukan inference sederhana dari model yang telah disimpan di wandb.
3.  `train_model.py`: Python script untuk melakukan otomasi model development. Metrics tracking dan model versioning bisa dilihat pada [Weight & Biases (Wandb)](https://wandb.ai/luthfi-organization/trash-classification?nw=nwuserluthfi507).
   
## How to Reproduce
Clone repository,
```bash
git clone https://github.com/Luthfi507/trash-classification.git
```
Install dependencies,
```bash
pip install -r requirements.txt
```
Buat `models` directory
```bash
mkdir -p `./models/`
```
Run `train_mode.py`,
```bash
python train_model.py
```

## Pre-trained Models
Model hasil pelatihan dapat digunakan dari [Huggingface repository](https://huggingface.co/luthfi507/trash-classification).