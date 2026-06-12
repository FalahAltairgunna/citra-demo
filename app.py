import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import os
import glob
import shutil
import random
import json
import zipfile
import io
import time

MODEL_DIR = "models"
MODEL_PATHS = {
    "EfficientNet-B0": os.path.join(MODEL_DIR, "best_EfficientNet-B0.pth"),
    "MobileNetV3": os.path.join(MODEL_DIR, "best_MobileNetV3.pth"),
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🔥 Disaster Classification App",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-label { color: #aaa; font-size: 0.8rem; }
    .metric-value { color: #fff; font-size: 1.6rem; font-weight: 700; }
    .stAlert { border-radius: 8px; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES = ['Non_Damage_Building', 'Non_Damage_Wildlife', 'Urban_Fire', 'Wild_Fire']
DATASET_DIR = "custom_fire_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in {
    "dataset_ready": False,
    "dataset_dir": DATASET_DIR,
    "model_eff": None,
    "model_mob": None,
    "history_eff": None,
    "history_mob": None,
    "eff_labels": None,
    "eff_preds": None,
    "mob_labels": None,
    "mob_preds": None,
    "class_names": CLASS_NAMES,
    "training_done": False,
    "val_dataset": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_model(arch, num_classes, pretrained=True):
    if arch == "EfficientNet-B0":
        model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
    else:
        model = timm.create_model('mobilenetv3_small_100', pretrained=pretrained, num_classes=num_classes)
    return model.to(DEVICE)


def safe_load_checkpoint(file_or_path):
    try:
        checkpoint = torch.load(file_or_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(file_or_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise ValueError("File .pth tidak berisi state_dict model yang valid.")

    return {k.replace("module.", ""): v for k, v in checkpoint.items()}


def load_uploaded_pth(uploaded_file, arch, num_classes):
    model = build_model(arch, num_classes, pretrained=False)
    checkpoint_bytes = io.BytesIO(uploaded_file.getvalue())
    state_dict = safe_load_checkpoint(checkpoint_bytes)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_model_from_path(arch, checkpoint_path, num_classes):
    model = build_model(arch, num_classes, pretrained=False)
    state_dict = safe_load_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def auto_load_models_from_folder():
    loaded = []
    errors = []

    for arch, checkpoint_path in MODEL_PATHS.items():
        state_key = "model_eff" if arch == "EfficientNet-B0" else "model_mob"
        if st.session_state[state_key] is not None:
            loaded.append(arch)
            continue

        if not os.path.exists(checkpoint_path):
            continue

        try:
            st.session_state[state_key] = load_model_from_path(
                arch,
                checkpoint_path,
                len(st.session_state.class_names),
            )
            loaded.append(arch)
        except Exception as exc:
            errors.append(f"{arch}: {exc}")

    st.session_state.training_done = bool(
        st.session_state.model_eff is not None or st.session_state.model_mob is not None
    )
    return loaded, errors


def evaluate_model_on_dataset(model, dataset_dir, batch_size=32):
    dataset = datasets.ImageFolder(dataset_dir, transform=VAL_TRANSFORMS)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    labels, preds = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            batch_preds = outputs.argmax(1).cpu().numpy()
            preds.extend(batch_preds)
            labels.extend(targets.numpy())

    return labels, preds, dataset.classes


def prepare_dataset_from_zip(zip_path, target_dir):
    """Extract zip & restructure into target_dir/ClassName/"""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("_raw_dataset")

    class_map = {
        'Urban_Fire':             ['Urban_Fire'],
        'Wild_Fire':              ['Wild_Fire'],
        'Non_Damage_Building':    ['Non_Damage_Buildings_Street', 'Non_Damage_Building'],
        'Non_Damage_Wildlife':    ['Non_Damage_Wildlife_Forest', 'Non_Damage_Wildlife'],
    }

    counts = {}
    for class_name, source_variants in class_map.items():
        all_images = []
        for variant in source_variants:
            pattern = f'_raw_dataset/**/*{variant}*'
            for d in glob.glob(pattern, recursive=True):
                if os.path.isdir(d):
                    all_images += glob.glob(f'{d}/*.jpg') + glob.glob(f'{d}/*.png') + glob.glob(f'{d}/*.jpeg')

        if not all_images:
            continue

        limit = 500 if "Non_Damage" in class_name else 10000
        selected = random.sample(all_images, min(limit, len(all_images)))

        dest = os.path.join(target_dir, class_name)
        os.makedirs(dest, exist_ok=True)
        for img_path in selected:
            shutil.copy(img_path, dest)
        counts[class_name] = len(selected)

    shutil.rmtree("_raw_dataset", ignore_errors=True)
    return counts


def train_one_model(model, optimizer, scheduler, name, train_loader, val_loader, class_names, epochs, progress_bar, status_text):
    best_val_acc = 0.0
    best_path = f"best_{name.replace(' ', '_')}.pth"
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    all_labels, all_preds = [], []

    for epoch in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_loss = run_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        ep_labels, ep_preds = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                out = model(inputs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                _, pred = out.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
                ep_preds.extend(pred.cpu().numpy())
                ep_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        all_labels, all_preds = ep_labels, ep_preds

        progress_bar.progress((epoch + 1) / epochs)
        status_text.markdown(
            f"**{name}** — Epoch {epoch+1}/{epochs} | "
            f"Train Acc: `{train_acc:.1f}%` | Val Acc: `{val_acc:.1f}%` | "
            f"Best: `{best_val_acc:.1f}%`"
        )

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return model, history, all_labels, all_preds, best_path


def plot_learning_curves(history_eff, history_mob):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#0e1117')

    pairs = [
        (axes[0, 0], history_eff, 'EfficientNet-B0', 'train_acc', 'val_acc', 'Akurasi (%)', '#4CAF50', '#81C784'),
        (axes[0, 1], history_mob, 'MobileNetV3', 'train_acc', 'val_acc', 'Akurasi (%)', '#2196F3', '#64B5F6'),
        (axes[1, 0], history_eff, 'EfficientNet-B0', 'train_loss', 'val_loss', 'Loss', '#F44336', '#EF9A9A'),
        (axes[1, 1], history_mob, 'MobileNetV3', 'train_loss', 'val_loss', 'Loss', '#FF9800', '#FFCC80'),
    ]

    for ax, hist, title, k1, k2, ylabel, c1, c2 in pairs:
        epochs = range(1, len(hist[k1]) + 1)
        ax.set_facecolor('#1e1e2e')
        ax.plot(epochs, hist[k1], label='Train', color=c1, linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, hist[k2], label='Validasi', color=c2, linewidth=2, marker='s', markersize=4, linestyle='--')
        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', color='#aaa')
        ax.set_ylabel(ylabel, color='#aaa')
        ax.tick_params(colors='#aaa')
        ax.legend(facecolor='#2a2a3e', labelcolor='white')
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    plt.tight_layout()
    return fig


def plot_confusion_matrix(labels, preds, class_names, title):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e2e')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c.replace('_', '\n') for c in class_names],
                yticklabels=[c.replace('_', '\n') for c in class_names],
                ax=ax, annot_kws={"size": 11},
                linewidths=0.5, linecolor='#333')
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Prediksi', color='#aaa', fontsize=10)
    ax.set_ylabel('Ground Truth', color='#aaa', fontsize=10)
    ax.tick_params(colors='#aaa')
    plt.tight_layout()
    return fig


def predict_image(model, img_pil, class_names):
    model.eval()
    tensor = VAL_TRANSFORMS(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    pred_idx = np.argmax(probs)
    return pred_idx, probs


def get_gradcam(model, img_pil, arch):
    """Simple Grad-CAM without external library dependency."""
    model.eval()
    img_tensor = VAL_TRANSFORMS(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)

    # choose target layer
    if arch == "EfficientNet-B0":
        target_layer = model.conv_head
    else:
        # MobileNetV3 small: last block
        target_layer = model.blocks[-1]

    activation = {}
    gradient = {}

    def forward_hook(module, input, output):
        activation['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradient['value'] = grad_out[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    out = model(img_tensor)
    pred_class = out.argmax(dim=1).item()
    model.zero_grad()
    out[0, pred_class].backward()

    fh.remove()
    bh.remove()

    grads = gradient['value']        # (1, C, H, W)
    acts  = activation['value']      # (1, C, H, W)
    weights = grads.mean(dim=[2, 3], keepdim=True)  # GAP
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    if cam.ndim == 0:
        cam = np.zeros((7, 7))

    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    # Overlay
    img_np = np.array(img_pil.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255.0

    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    overlay = 0.5 * img_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay, pred_class


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
loaded_models, load_errors = auto_load_models_from_folder()

with st.sidebar:
    st.markdown("### 🔥 Disaster Classifier")
    st.markdown("*Evaluasi & Prediksi Grad-CAM*")
    st.divider()

    page = st.radio("Navigasi", [
        "📊 Evaluasi",
        "🔍 Prediksi & Grad-CAM",
    ], label_visibility="collapsed")

    st.divider()
    st.caption(f"Device: `{DEVICE}`")
    if loaded_models:
        st.success("✅ Model aktif")
        for model_name in loaded_models:
            st.caption(f"- {model_name}")
    else:
        st.warning("⚠️ Tidak ada model .pth di folder models")
    if load_errors:
        with st.expander("Detail error model"):
            for error in load_errors:
                st.code(error, language="text")

# ─────────────────────────────────────────────
# PAGE: BERANDA
# ─────────────────────────────────────────────
if page == "🏠 Beranda":
    st.markdown('<div class="main-title">🔥 Disaster Image Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evaluasi Komparatif EfficientNet-B0 & MobileNetV3 dengan Interpretasi Grad-CAM</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Kelas</div>
            <div class="metric-value">4</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">2</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Input Size</div>
            <div class="metric-value">224²</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">XAI</div>
            <div class="metric-value">Grad-CAM</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Tentang Proyek")
    st.markdown("""
Aplikasi ini merupakan GUI interaktif untuk penelitian **klasifikasi citra bencana kebakaran** menggunakan dua arsitektur *Edge-AI*:

| Komponen | Detail |
|---|---|
| **Dataset** | Comprehensive Disaster Dataset (CDD) |
| **Kelas** | Urban Fire, Wild Fire, Non-Damage Building, Non-Damage Wildlife |
| **Model 1** | EfficientNet-B0 (via `timm`) |
| **Model 2** | MobileNetV3 Small (via `timm`) |
| **Optimizer** | Adam + Weight Decay (1e-4) |
| **Scheduler** | CosineAnnealingLR |
| **XAI** | Grad-CAM (built-in, tanpa library eksternal) |

### 🗺️ Alur Penggunaan
1. **📂 Dataset** → Upload file ZIP dataset dan siapkan data
2. **🚀 Training** → Latih EfficientNet-B0 dan/atau MobileNetV3
3. **📊 Evaluasi** → Lihat confusion matrix & classification report
4. **🔍 Prediksi & Grad-CAM** → Upload gambar baru, lihat prediksi + visualisasi Grad-CAM
    """)

# ─────────────────────────────────────────────
# PAGE: DATASET
# ─────────────────────────────────────────────
elif page == "📂 Dataset":
    st.title("📂 Persiapan Dataset")

    tab1, tab2 = st.tabs(["📤 Upload & Siapkan", "🖼️ Eksplorasi Gambar"])

    with tab1:
        st.markdown("Upload file ZIP dataset kamu. Struktur yang didukung:")
        st.code("""
Comprehensive Disaster Dataset(CDD)/
├── Fire_Disaster/
│   ├── Urban_Fire/
│   └── Wild_Fire/
└── Non_Damage/
    ├── Non_Damage_Buildings_Street/
    └── Non_Damage_Wildlife_Forest/
        """)

        uploaded_zip = st.file_uploader("Upload Dataset ZIP", type=['zip'])

        if uploaded_zip:
            with open("_uploaded_dataset.zip", "wb") as f:
                f.write(uploaded_zip.read())

            with st.spinner("⏳ Mengekstrak dan menyusun ulang dataset..."):
                try:
                    counts = prepare_dataset_from_zip("_uploaded_dataset.zip", DATASET_DIR)
                    st.session_state.dataset_ready = True
                    st.session_state.dataset_dir = DATASET_DIR
                    detected_classes = sorted([c for c in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, c))])
                    st.session_state.class_names = detected_classes

                    st.success("✅ Dataset berhasil disiapkan!")
                    total = sum(counts.values())
                    cols = st.columns(len(counts))
                    for i, (cls, cnt) in enumerate(counts.items()):
                        with cols[i]:
                            st.metric(cls.replace('_', ' '), cnt)
                    st.metric("Total Gambar", total)
                except Exception as e:
                    st.error(f"❌ Gagal memproses dataset: {e}")
        else:
            st.info("Belum ada ZIP yang diupload.")

        if st.session_state.dataset_ready and os.path.exists(DATASET_DIR):
            st.divider()
            st.subheader("📊 Distribusi Kelas")
            cls_names = st.session_state.class_names
            counts_dict = {c: len(os.listdir(os.path.join(DATASET_DIR, c))) for c in cls_names if os.path.isdir(os.path.join(DATASET_DIR, c))}

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e1e2e')
            bars = ax.bar(counts_dict.keys(), counts_dict.values(),
                         color=['#FF4B2B', '#FF9800', '#4CAF50', '#2196F3'], edgecolor='#333', linewidth=0.8)
            ax.set_title('Distribusi Gambar per Kelas', color='white', fontsize=13)
            ax.tick_params(colors='#aaa')
            ax.set_xticklabels([k.replace('_', '\n') for k in counts_dict.keys()], color='#aaa')
            for bar, v in zip(bars, counts_dict.values()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(v),
                       ha='center', va='bottom', color='white', fontsize=11)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        if not st.session_state.dataset_ready:
            st.warning("Siapkan dataset terlebih dahulu di tab 'Upload & Siapkan'.")
        else:
            cls_names = st.session_state.class_names
            selected_class = st.selectbox("Pilih Kelas", cls_names)
            num_show = st.slider("Jumlah gambar yang ditampilkan", 4, 20, 8)

            class_path = os.path.join(DATASET_DIR, selected_class)
            all_imgs = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            sample_imgs = random.sample(all_imgs, min(num_show, len(all_imgs)))

            cols_per_row = 4
            for row_start in range(0, len(sample_imgs), cols_per_row):
                row_imgs = sample_imgs[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, fname in zip(cols, row_imgs):
                    img = Image.open(os.path.join(class_path, fname))
                    col.image(img, caption=fname[:20], use_container_width=True)

            st.caption(f"Menampilkan {len(sample_imgs)} dari {len(all_imgs)} gambar di kelas **{selected_class}**")

# ─────────────────────────────────────────────
# PAGE: TRAINING
# ─────────────────────────────────────────────
elif page == "🚀 Training":
    st.title("🚀 Training Model")

    if not st.session_state.dataset_ready:
        st.error("❌ Dataset belum disiapkan! Pergi ke halaman **📂 Dataset** terlebih dahulu.")
        st.stop()

    with st.expander("⚙️ Konfigurasi Training", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.slider("Jumlah Epoch", 3, 30, 15)
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        with col2:
            lr = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001)
            weight_decay = st.select_slider("Weight Decay", [1e-5, 1e-4, 1e-3], value=1e-4)
        with col3:
            train_eff = st.checkbox("Latih EfficientNet-B0", value=True)
            train_mob = st.checkbox("Latih MobileNetV3", value=True)
            val_split = st.slider("Validasi Split (%)", 10, 30, 20)

    if st.button("▶️ Mulai Training", type="primary", use_container_width=True):
        if not train_eff and not train_mob:
            st.error("Pilih minimal satu model untuk dilatih!")
            st.stop()

        cls_names = st.session_state.class_names
        num_classes = len(cls_names)

        # Prepare data
        with st.spinner("📦 Memuat dataset..."):
            full_dataset = datasets.ImageFolder(DATASET_DIR, transform=TRAIN_TRANSFORMS)
            actual_classes = full_dataset.classes
            st.session_state.class_names = actual_classes
            num_classes = len(actual_classes)

            train_size = int((1 - val_split / 100) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            val_ds.dataset.transform = VAL_TRANSFORMS

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            st.session_state.val_dataset = val_ds
            st.info(f"Train: {train_size} | Val: {val_size} | Kelas: {actual_classes}")

        models_to_train = []
        if train_eff:
            models_to_train.append(("EfficientNet-B0", "eff"))
        if train_mob:
            models_to_train.append(("MobileNetV3", "mob"))

        for model_name, key in models_to_train:
            st.subheader(f"🤖 Melatih {model_name}")
            progress = st.progress(0)
            status = st.empty()

            model = build_model(model_name, num_classes)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            trained_model, history, all_labels, all_preds, ckpt_path = train_one_model(
                model, optimizer, scheduler, model_name,
                train_loader, val_loader, actual_classes,
                epochs, progress, status
            )

            st.session_state[f"model_{key}"] = trained_model
            st.session_state[f"history_{key}"] = history
            st.session_state[f"{key}_labels"] = all_labels
            st.session_state[f"{key}_preds"] = all_preds

            best_val = max(history['val_acc'])
            st.success(f"✅ {model_name} selesai! Best Val Acc: **{best_val:.1f}%**")

            # Download button for model weights
            with open(ckpt_path, "rb") as f:
                st.download_button(
                    f"⬇️ Download bobot {model_name} (.pth)",
                    data=f,
                    file_name=ckpt_path,
                    mime="application/octet-stream",
                    key=f"dl_{key}"
                )

        st.session_state.training_done = True
        st.balloons()
        st.success("🎉 Semua training selesai! Pergi ke **📊 Evaluasi** untuk melihat hasil.")

# ─────────────────────────────────────────────
# PAGE: EVALUASI
# ─────────────────────────────────────────────
elif page == "📊 Evaluasi":
    st.title("📊 Evaluasi Model")

    has_eff = st.session_state.model_eff is not None
    has_mob = st.session_state.model_mob is not None

    if not has_eff and not has_mob:
        st.warning("Belum ada model aktif. Pastikan file `.pth` tersedia di folder `models`.")
        st.stop()

    st.subheader("Model yang Dimuat")
    model_rows = []
    for model_name, is_loaded in [
        ("EfficientNet-B0", has_eff),
        ("MobileNetV3", has_mob),
    ]:
        model_rows.append({
            "Model": model_name,
            "Status": "Aktif" if is_loaded else "Tidak ditemukan",
            "Path": MODEL_PATHS[model_name],
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

    st.info(
        "Aplikasi ini sekarang memakai bobot `.pth` dari folder `models` dan tidak melakukan training ulang. "
        "Confusion matrix/classification report hanya bisa dihitung jika folder dataset validasi tersedia."
    )

    if not os.path.exists(DATASET_DIR):
        st.warning(
            f"Folder `{DATASET_DIR}` tidak ditemukan. Evaluasi metrik dilewati, tetapi prediksi dan Grad-CAM tetap bisa digunakan."
        )
        st.stop()

    if st.button("Hitung Evaluasi dari Dataset Lokal", type="primary"):
        models_to_eval = []
        if has_eff:
            models_to_eval.append(("EfficientNet-B0", st.session_state.model_eff))
        if has_mob:
            models_to_eval.append(("MobileNetV3", st.session_state.model_mob))

        for model_name, model in models_to_eval:
            with st.spinner(f"Mengevaluasi {model_name}..."):
                labels, preds, dataset_classes = evaluate_model_on_dataset(model, DATASET_DIR)

            st.markdown(f"### {model_name}")
            report = classification_report(
                labels,
                preds,
                target_names=dataset_classes,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

            fig = plot_confusion_matrix(labels, preds, dataset_classes, model_name)
            st.pyplot(fig)
            plt.close(fig)

# ─────────────────────────────────────────────
# PAGE: PREDIKSI & GRAD-CAM
# ─────────────────────────────────────────────
elif page == "🔍 Prediksi & Grad-CAM":
    st.title("🔍 Prediksi Gambar & Grad-CAM")

    has_eff = st.session_state.model_eff is not None
    has_mob = st.session_state.model_mob is not None

    if not has_eff and not has_mob:
        st.warning(
            "Belum ada model aktif. Pastikan file `.pth` ada di folder `models`: "
            "`best_MobileNetV3.pth` atau `best_EfficientNet-B0.pth`."
        )
        st.stop()

    st.success("Model otomatis dimuat dari folder `models`.")

    col_upload, col_config = st.columns([2, 1])

    with col_upload:
        uploaded_img = st.file_uploader("Upload Gambar untuk Diprediksi", type=['jpg', 'jpeg', 'png'])

    with col_config:
        active_models = []
        if has_eff:
            active_models.append("EfficientNet-B0")
        if has_mob:
            active_models.append("MobileNetV3")

        selected_models = st.multiselect("Model yang digunakan", active_models, default=active_models)
        show_gradcam = st.checkbox("Tampilkan Grad-CAM", value=True)

    if uploaded_img:
        img_pil = Image.open(uploaded_img).convert("RGB")
        cls_names = st.session_state.class_names

        st.divider()

        # Show original image
        col_img, col_results = st.columns([1, 2])
        with col_img:
            st.image(img_pil, caption="Gambar Original", use_container_width=True)

        with col_results:
            st.subheader("📊 Hasil Prediksi")

            for model_name in selected_models:
                model = st.session_state.model_eff if model_name == "EfficientNet-B0" else st.session_state.model_mob
                pred_idx, probs = predict_image(model, img_pil, cls_names)

                pred_label = cls_names[pred_idx]
                confidence = probs[pred_idx] * 100

                label_color = "🔴" if "Fire" in pred_label else "🟢"
                st.markdown(f"**{model_name}**: {label_color} `{pred_label.replace('_', ' ')}` — **{confidence:.1f}%** keyakinan")

                # Bar chart of probs
                fig, ax = plt.subplots(figsize=(6, 2.5))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#1e1e2e')
                colors = ['#FF4B2B' if i == pred_idx else '#444' for i in range(len(cls_names))]
                bars = ax.barh([c.replace('_', '\n') for c in cls_names], probs * 100, color=colors, edgecolor='#222')
                ax.set_xlim(0, 105)
                for bar, p in zip(bars, probs * 100):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                           f'{p:.1f}%', va='center', color='white', fontsize=9)
                ax.tick_params(colors='#aaa', labelsize=8)
                ax.set_xlabel('Probabilitas (%)', color='#aaa', fontsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#333')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Grad-CAM section
        if show_gradcam:
            st.divider()
            st.subheader("🔥 Visualisasi Grad-CAM (Explainable AI)")
            st.caption("Area yang dihighlight menunjukkan region yang paling berpengaruh dalam keputusan model.")

            gcam_cols = st.columns(len(selected_models) + 1)
            with gcam_cols[0]:
                st.image(img_pil.resize((224, 224)), caption="Original", use_container_width=True)

            for i, model_name in enumerate(selected_models):
                model = st.session_state.model_eff if model_name == "EfficientNet-B0" else st.session_state.model_mob
                arch_key = "EfficientNet-B0" if model_name == "EfficientNet-B0" else "MobileNetV3"

                with st.spinner(f"Menghitung Grad-CAM untuk {model_name}..."):
                    try:
                        overlay, pred_idx = get_gradcam(model, img_pil, arch_key)
                        overlay_img = (overlay * 255).astype(np.uint8)

                        with gcam_cols[i + 1]:
                            st.image(overlay_img, caption=f"Grad-CAM: {model_name}", use_container_width=True)
                            st.caption(f"Prediksi: {cls_names[pred_idx].replace('_', ' ')}")
                    except Exception as e:
                        with gcam_cols[i + 1]:
                            st.error(f"Grad-CAM gagal: {e}")
    else:
        # Demo with dataset images
        if st.session_state.dataset_ready:
            st.info("💡 Atau coba dengan gambar dari dataset:")
            cls_names = st.session_state.class_names
            demo_class = st.selectbox("Pilih kelas untuk demo", cls_names)
            class_path = os.path.join(DATASET_DIR, demo_class)
            imgs = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if imgs and st.button("🎲 Ambil Gambar Acak dari Dataset"):
                rand_img = random.choice(imgs)
                demo_pil = Image.open(os.path.join(class_path, rand_img))
                st.image(demo_pil, caption=f"Gambar dari kelas: {demo_class}", width=300)
                st.info("Upload gambar ini atau gambar lain di atas untuk memprediksinya.")
        else:
            st.info("Upload gambar di atas untuk mulai prediksi.")
