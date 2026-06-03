# Bean Leaf Disease Classification

This project is a deep learning-based image classification tool that detects diseases in bean plant leaves. It categorizes images into three classes:
- **Angular leaf spot**
- **Bean rust**
- **Healthy**
- <img width="2952" height="1752" alt="image" src="https://github.com/user-attachments/assets/d9858b77-56cc-43a9-8029-7f55575381f7" />


## Features
- **Model Training Notebook**: A comprehensive Jupyter Notebook (`bean_disease_classifications.ipynb`) covering data loading, augmentation, and training of three distinct deep learning architectures.
- **Multiple Models**: 
  - Custom Baseline CNN
  - Improved CNN (with Dropout and adjusted layers)
  - ResNet18 (Fine-tuned Transfer Learning)
- **Interactive Web UI**: A web interface built using [Gradio](https://gradio.app/) where users can upload an image of a leaf and get real-time predictions from the models.

## Tech Stack
- **Python** (PyTorch, Torchvision)
- **HuggingFace Datasets**
- **Gradio** (Web UI)
- **Matplotlib & Seaborn** (Data visualization and result analysis)
- **Scikit-learn** (Evaluation metrics like confusion matrices and ROC curves)

## Project Structure
- `app.py`: The Gradio web application script.
- `bean_disease_classifications.ipynb`: The main notebook for training and evaluating the models.
- `*.pth`: PyTorch model weights (e.g., `baseline_cnn.pth`, `resnet18_ft.pth`).
- `classes.json`: Contains the category names.
- `*.png`: Saved evaluation curves and graphs (accuracy, loss, ROC, confusion matrix).

## How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install torch torchvision pillow gradio huggingface_hub datasets scikit-learn pandas matplotlib seaborn
```

### 2. (Optional) Train the Models
If you want to train the models yourself:
1. Open `bean_disease_classifications.ipynb`.
2. Run all the cells to download the HuggingFace dataset, train the CNNs, and save the `.pth` model files.

### 3. Run the Gradio App Web Interface
To start the web application, simply execute:
```bash
python3 app.py
```
This will launch a local web server (usually at `http://127.0.0.1:7860`). Click the link in your terminal to open the UI in your web browser.

## 📈 Results
The models were evaluated based on their test accuracy, F1-scores, and per-class performance. Transfer learning using ResNet18 generally yields the most robust predictions on the validation and test datasets.

### Model Accuracy
<img width="2100" height="1500" alt="image" src="https://github.com/user-attachments/assets/4b1b2be9-5f04-4aee-9ab8-3d54eb3c4ad4" />


