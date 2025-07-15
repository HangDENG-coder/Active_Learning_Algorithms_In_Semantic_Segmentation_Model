
# 🔬 Active Learning for Urine Sediment Classification

This repository implements an uncertainty-driven active learning pipeline for improving model performance on urine sediment classification while minimizing manual labeling effort. The framework combines uncertainty estimation, sample selection, and human-in-the-loop annotation to target the most informative data for labeling. 

Please be noted, out of intellectual property protection, here we are not sharing the model and data related files, but only the active learning algorithms.

## 📜 Overview

Microscopy-based urine sediment examination is essential for diagnosing kidney, liver, and metabolic disorders. However, changes in imaging settings can reduce model generalizability. This project addresses that challenge by:

- Quantifying prediction uncertainty using ensemble models.
- Selecting and labeling uncertain samples through a GUI-assisted workflow.
- Iteratively retraining a CNN classifier on the most informative examples.

This system was deployed in collaboration with Phast Diagnostics to address real-world domain shift scenarios.

---

## 📁 Project Structure

```
Active_learning/
│
├── Active_learning_main_code.py     # Entry point for training + active loop
├── acquisition_fuc.py               # Uncertainty metric functions (entropy, BALD, etc.)
├── model_prediction.py              # Model inference utilities
├── obj_active_learning.py           # Active learning loop and batch management
├── visualization.py                 # Plotting and uncertainty maps
├── GUI/                             # GUI for human-in-the-loop annotation
│   └── gui_annotation.py
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

---

## 🧠 Key Features

- **Ensemble CNN classifier**: Predicts pixel-wise labels for urinary particles.
- **Uncertainty metrics**: Includes entropy, variation ratios, BALD, standard deviation, and least confidence.
- **Sample selection**: Actively identifies high-uncertainty samples for annotation.
- **Annotation GUI**: Allows domain experts to verify and label selected samples interactively.
- **Adaptive metric selection**: Chooses the most effective uncertainty measure for a given batch.
- **Real-world validation**: Evaluated using microscopy images from a clinical lab.

---

## 🖥️ Installation

```bash
git clone https://github.com/your-org/Active_learning.git
cd Active_learning
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the active learning loop
```bash
python Active_learning_main_code.py --config config.yaml
```

### Visualize uncertainty map
```bash
python visualization.py --sample_id 12
```

### Launch the annotation GUI
```bash
python gui_annotation.py
```

---

## 📊 Evaluation Highlights

- Entropy-based sampling achieved **10% higher accuracy** than random selection.
- Model uncertainty is highest at **cell boundaries**—regions with low contrast or ambiguous morphology.
- GUI workflow increased annotation efficiency by focusing expert effort on uncertain cases.

---

## 🔒 License

This project is licensed under the [MIT License](LICENSE).

---

## ✍️ Citation

If you use this code or method, please cite:

> Hang Deng, Vivek Venkatachalam. *Algorithms for 3D Neuron Tracking and Identification in \textit{C.~elegans}, and Key Points Detection.*
>
> [Thesis Chapter 4, 2025]

---




