import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import json
import os

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/Users/rahulchaudhary/Documents/beansproject"
CLASSES    = json.load(open(os.path.join(BASE_DIR, "classes.json")))
NUM_CLS    = len(CLASSES)
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Model Definitions ────────────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 32 * 32, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1     = nn.Linear(128 * 16 * 16, 256)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ── Load Models ──────────────────────────────────────────────────────────────
def load_model(cls, path, **kwargs):
    m = cls(**kwargs)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m

baseline = load_model(BaselineCNN, os.path.join(BASE_DIR, "baseline_cnn.pth"),  num_classes=NUM_CLS)
improved = load_model(ImprovedCNN,  os.path.join(BASE_DIR, "improved_cnn.pth"), num_classes=NUM_CLS)

resnet = models.resnet18(pretrained=False)
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.layer4.parameters():
    param.requires_grad = True
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLS)
resnet.load_state_dict(torch.load(os.path.join(BASE_DIR, "resnet18_ft.pth"), map_location=DEVICE))
resnet.to(DEVICE)
resnet.eval()

# ── Transforms ───────────────────────────────────────────────────────────────
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

cnn_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

res_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ── Prediction ────────────────────────────────────────────────────────────────
THRESHOLD = 0.70
ENTROPY_THRESHOLD = 0.8

def predict(image: Image.Image):
    img_cnn = cnn_tf(image).unsqueeze(0).to(DEVICE)
    img_res = res_tf(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        p_base = torch.softmax(baseline(img_cnn), dim=1)[0]
        p_imp  = torch.softmax(improved(img_cnn),  dim=1)[0]
        p_res  = torch.softmax(resnet(img_res),    dim=1)[0]

    def fmt(p):
        return {CLASSES[i]: float(p[i]) for i in range(NUM_CLS)}

    out_base = fmt(p_base)
    out_imp  = fmt(p_imp)
    out_res  = fmt(p_res)

    return out_base, out_imp, out_res

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Bean Disease Classifier",
    theme=gr.themes.Soft(),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif !important;
    }

    body {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .gradio-container {
        max-width: 1400px !important;
    }

    h1 {
        font-weight: 700 !important;
        letter-spacing: -1px;
    }

    p, label, span, button, textarea, input {
        font-weight: 500 !important;
    }
    """
) as demo:

    gr.Markdown("# 🍃 Bean Disease Classifier")
    gr.Markdown("Upload a bean leaf image and compare predictions from all 3 models instantly.")

    with gr.Row():
        img_input = gr.Image(
            type="pil",
            label="📷Upload Bean Leaf"
        )

        predict_btn = gr.Button(
            "🔍 Predict",
            variant="primary",
            scale=0
        )

    with gr.Row():

        out1 = gr.Label(
            num_top_classes=3,
            label="Baseline CNN"
        )

        out2 = gr.Label(
            num_top_classes=3,
            label="Improved CNN"
        )

        out3 = gr.Label(
            num_top_classes=3,
            label="ResNet18"
        )

    predict_btn.click(
        fn=predict,
        inputs=img_input,
        outputs=[out1, out2, out3]
    )

    img_input.change(
        fn=predict,
        inputs=img_input,
        outputs=[out1, out2, out3]
    )

    gr.Markdown(
        "**Classes:** `Angular leaf spot` · `Bean rust` · `Healthy`"
    )

demo.launch()