# Composer Classification Using Deep Learning

**Project By: Christopher Mendoza, Greg Moore, and Alexander Padin**

MSAAI-511-Group-3 | July 2025  
Graduate Students | Masters in Applied Artificial Intelligence | University of San Diego

---

## Introduction

Music is a form of art that is ubiquitous and has a rich history. Different composers have created music with their unique styles and compositions. However, identifying the composer of a particular piece of music can be a challenging task, especially for novice musicians or listeners. This project uses deep learning techniques to accurately identify the composer of a given piece of music.

---

## Objective

The primary objective of this project is to develop a deep learning model that can predict the composer of a given musical score. The project explores two deep learning approaches:

- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)

---

## Dataset

The dataset includes musical scores from various classical composers such as:

- Johann Sebastian Bach
- Ludwig van Beethoven
- Frédéric Chopin
- Wolfgang Amadeus Mozart
- Franz Schubert

The dataset consists of:

- MIDI files
- Sheet music (optional)

Each score is labeled with the correct composer to support supervised learning.

Dateset should be retrieved from: https://www.kaggle.com/datasets/blanderbuss/midi-classic-music

---

## Methodology

1. **Data Collection**  
   A curated dataset of classical music scores is provided.

2. **Data Pre-processing**  
   Convert musical scores to MIDI format and apply data augmentation techniques.

3. **Feature Extraction**  
   Extract musical features such as notes, chords, rhythm, and tempo using music analysis tools.

4. **Model Building**  
   Design and implement deep learning models using LSTM and CNN architectures.

5. **Model Training**  
   Train models on the processed and feature-rich dataset.

6. **Model Evaluation**  
   Evaluate using accuracy, precision, recall, and F1-score.

7. **Model Optimization**  
   Fine-tune hyperparameters and apply regularization techniques to improve model performance.

---

## Technologies Used

- Python
- TensorFlow / Keras
- PyTorch (optional)
- music21, pretty_midi
- Google Colab
- GitHub

---

## Repository Structure (Planned)

```
AAI-511-04-Group-3/
├── data/
│   └── midi_files/
├── notebooks/
│   └── lstm_model.ipynb
│   └── cnn_model.ipynb
├── models/
├── utils/
├── README.md
└── requirements.txt
```

---

## Future Work

- Integrate Transformer-based models for improved temporal modeling
- Develop a web interface for real-time composer identification
- Expand the dataset to include modern composers

---

## License

This project is for educational purposes only as part of the MSAAI program at the University of San Diego.
