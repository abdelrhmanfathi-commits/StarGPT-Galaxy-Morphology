# StarGPT: Galaxy Morphology Classification with Zoobot

**ECCE 635/787 Deep Learning Systems Design Project** **Department of Electrical Engineering & Computer Science**

## 👥 Group Members
1. Maryam Al Mesmari (100067995)
2. Nada Kawde (100066960)
3. Mohammed Ali Alyafeai (100067967)
4. Abdelrhman Fathi Abdelfatah (100065777)

## 🚀 Project Overview
StarGPT is a deep learning framework designed to automate the morphological classification of galaxies. Utilizing the **Galaxy Zoo 2** dataset and the pre-trained **Zoobot (ConvNeXt-Base)** encoder, this project classifies galaxies into 6 distinct morphological types:
- Smooth
- Edge-on
- Barred Spiral
- Unbarred Spiral
- Featured
- Artifact

We investigate the impact of **Class Imbalance** in astronomical data and propose solutions using **Focal Loss** and **Oversampling**.

## 📂 Repository Structure
- `notebooks/`: Contains the Jupyter Notebooks used for training and ablation studies.
  - `1_Baseline_Training.ipynb`: The main run on 172k samples (50 epochs).
  - `2_Ablation_Study.ipynb`: Comparative runs using Focal Loss and Oversampling.
- `results/`: Contains generated plots (Confusion Matrices, AUROC curves).
- `requirements.txt`: Python dependencies.

## 📊 Dataset
We utilize the **Galaxy Zoo 2 (GZ2)** dataset hosted on Hugging Face.
- **Source:** [mwalmsley/gz2](https://huggingface.co/datasets/mwalmsley/gz2)
- **Size:** ~240,000 images (Filtered down to 172k for training).
- **Note:** The dataset is streamed directly during training; no local download is required.

## 🛠️ How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt