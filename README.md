# ⚡ SmartGrid Insight: Power System Event Classification & Blockchain-based P2P Energy Trading

This project presents a complete pipeline for **automated classification of power system events** using deep learning (1D CNN) and enables secure **peer-to-peer (P2P) energy trading** via blockchain smart contracts. It includes a Flask web app to visualize event classification, waveform analysis, and simulate secure energy transfers.

---

## 🚀 Highlights

- 🧠 Classifies 45 event types (faults, inrush, capacitor switching, etc.)
- 📊 Trained on **100K+ differential current waveforms**, with **94.98% test accuracy**
- 🔄 Fully scalable PyTorch + Spark-based preprocessing
- 🔐 Ethereum Smart Contract for decentralized energy transactions
- 🌐 Flask web interface for model demo and blockchain simulation

---

## 📁 Project Structure
├── app.py # Flask web app entry point
├── model_def.py # CNN architecture definition
├── flask_artifacts/ # Pretrained model, label maps, sample data
├── static/ # Images, plots, waveform visualizations
├── templates/ # HTML templates for UI
├── requirements.txt # Required Python packages
├── README.md # This file


---

## 🧠 Deep Learning Pipeline

- **Input**: 3-phase differential current waveform → shape `(3 × 726)`
- **Model**: 1D CNN with 3 convolutional blocks
- **Training**: 
  - Optimizer: Adam
  - Loss: CrossEntropy
  - Accuracy: 94.98% on test set
- **Data Volume**: 100,908 waveform samples across 45 classes

---

## 🔗 Blockchain Smart Contract

- **File**: `EnergyTrade.sol` (not in repo, used in deployment)
- **Platform**: Ethereum (tested on Ganache + Remix)
- **Core Functions**:
  - `addSurplusEnergy()`
  - `transferEnergy()`
  - `getEnergyBalance()`

> Uses Solidity and simulates decentralized trading within microgrid environments.

---

## 🌐 Web App Features

- Upload or choose sample waveform
- Visualize 3-phase signal
- Predict event class and display human-readable impact explanation
- Simulate energy trade between producer & consumer
  
<img width="651" height="536" alt="image" src="https://github.com/user-attachments/assets/b69629d9-015d-42f4-a561-a495307c31b2" />


---

## ⚙️ Setup & Run

### 1. Install Requirements
bash
pip install -r requirements.txt

###2. Launch Flask App
bash
python app.py

###3. Access Locally
Open your browser at: http://localhost:5000
Make sure flask_artifacts/ contains:
  best_model.pth
  label_map.json
  norm_stats.json
  sample_inputs/
