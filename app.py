from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import transforms
    TORCH_IMPORT_ERROR = None
except (ImportError, OSError) as exc:  # pragma: no cover - shown in the Streamlit UI
    torch = None
    nn = None
    optim = None
    DataLoader = None
    random_split = None
    transforms = None
    TORCH_IMPORT_ERROR = exc

    class Dataset:  # type: ignore[no-redef]
        pass

try:
    import timm
    TIMM_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - shown in the Streamlit UI
    timm = None
    TIMM_IMPORT_ERROR = exc


APP_DIR = Path(__file__).resolve().parent
RAW_DATASET_DIR = APP_DIR / "Comprehensive Disaster Dataset(CDD)"
MODEL_DIR = APP_DIR / "models"

SOURCE_CLASS_DIRS = {
    "Fire_Disaster/Urban_Fire": "Urban_Fire",
    "Fire_Disaster/Wild_Fire": "Wild_Fire",
    "Non_Damage/Non_Damage_Buildings_Street": "Non_Damage_Building",
    "Non_Damage/Non_Damage_Wildlife_Forest": "Non_Damage_Wildlife",
}
CLASS_NAMES = [
    "Non_Damage_Building",
    "Non_Damage_Wildlife",
    "Urban_Fire",
    "Wild_Fire",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_OPTIONS = {
    "MobileNetV3": {
        "timm_name": "mobilenetv3_small_100",
        "checkpoint": APP_DIR / "best_MobileNetV3.pth",
    },
    "EfficientNet-B0": {
        "timm_name": "efficientnet_b0",
        "checkpoint": APP_DIR / "best_EfficientNet-B0.pth",
    },
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_name: str

    @property
    def label(self) -> int:
        return CLASS_NAMES.index(self.class_name)


class DisasterImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, record.label


def configure_page() -> None:
    st.set_page_config(
        page_title="Klasifikasi Citra Kebakaran",
        page_icon="fire",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; max-width: 1180px; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
        .stButton > button { width: 100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def scan_dataset(dataset_dir: str) -> list[ImageRecord]:
    root = Path(dataset_dir)
    records: list[ImageRecord] = []

    if not root.exists():
        return records

    for relative_dir, target_name in SOURCE_CLASS_DIRS.items():
        folder = root / relative_dir
        if not folder.is_dir():
            continue

        for image_path in folder.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(ImageRecord(image_path, target_name))

    return records


def summarize_records(records: Iterable[ImageRecord]) -> pd.DataFrame:
    rows = [{"Kelas": record.class_name, "Jumlah Gambar": 1} for record in records]
    if not rows:
        return pd.DataFrame(columns=["Kelas", "Jumlah Gambar"])

    summary = pd.DataFrame(rows).groupby("Kelas", as_index=False).sum()
    all_classes = pd.DataFrame({"Kelas": CLASS_NAMES})
    return all_classes.merge(summary, on="Kelas", how="left").fillna(0)


def build_transforms(training: bool):
    if transforms is None:
        raise RuntimeError(
            "Package torch dan torchvision belum terpasang. "
            "Jalankan: pip install -r Streamlit/requirements.txt"
        )

    base = [
        transforms.Resize((224, 224)),
    ]
    if training:
        base.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
            ]
        )

    base.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(base)


def make_model(model_key: str, pretrained: bool = False) -> nn.Module:
    if torch is None:
        raise RuntimeError(
            "Package torch dan torchvision belum terpasang. "
            "Jalankan: pip install -r Streamlit/requirements.txt"
        )
    if timm is None:
        raise RuntimeError("Package timm belum terpasang. Jalankan: pip install timm")

    config = MODEL_OPTIONS[model_key]
    return timm.create_model(
        config["timm_name"],
        pretrained=pretrained,
        num_classes=len(CLASS_NAMES),
    )


def get_device() -> torch.device:
    if torch is None:
        raise RuntimeError(
            "Package torch belum terpasang. Jalankan: pip install -r Streamlit/requirements.txt"
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_checkpoint(model_key: str) -> Path | None:
    preferred = MODEL_OPTIONS[model_key]["checkpoint"]
    candidates = [
        preferred,
        MODEL_DIR / preferred.name,
        APP_DIR / preferred.name.replace("-", "_"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def load_model(model_key: str, checkpoint_path: str):
    if torch is None:
        raise RuntimeError(
            "Package torch belum terpasang. Jalankan: pip install -r Streamlit/requirements.txt"
        )

    device = get_device()
    model = make_model(model_key, pretrained=False)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def prepare_balanced_records(
    records: list[ImageRecord],
    max_non_damage: int,
    seed: int,
) -> list[ImageRecord]:
    rng = random.Random(seed)
    selected: list[ImageRecord] = []

    for class_name in CLASS_NAMES:
        class_records = [record for record in records if record.class_name == class_name]
        if class_name.startswith("Non_Damage") and len(class_records) > max_non_damage:
            class_records = rng.sample(class_records, max_non_damage)
        selected.extend(class_records)

    rng.shuffle(selected)
    return selected


def predict_image(image: Image.Image, model: nn.Module, device: torch.device) -> pd.DataFrame:
    transform = build_transforms(training=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    result = pd.DataFrame(
        {
            "Kelas": CLASS_NAMES,
            "Probabilitas": probabilities,
        }
    ).sort_values("Probabilitas", ascending=False)
    result["Probabilitas"] = result["Probabilitas"].map(lambda value: round(float(value), 4))
    return result


def train_model(
    records: list[ImageRecord],
    model_key: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_non_damage: int,
    use_pretrained: bool,
    seed: int,
):
    if torch is None or DataLoader is None or random_split is None or optim is None or nn is None:
        raise RuntimeError(
            "Package torch dan torchvision belum terpasang. "
            "Jalankan: pip install -r Streamlit/requirements.txt"
        )

    torch.manual_seed(seed)
    balanced_records = prepare_balanced_records(records, max_non_damage, seed)
    if len(balanced_records) < len(CLASS_NAMES):
        raise RuntimeError("Dataset belum cukup untuk training semua kelas.")

    train_transform = build_transforms(training=True)
    val_transform = build_transforms(training=False)
    train_dataset_full = DisasterImageDataset(balanced_records, transform=train_transform)
    val_dataset_full = DisasterImageDataset(balanced_records, transform=val_transform)

    train_size = int(0.8 * len(balanced_records))
    val_size = len(balanced_records) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        range(len(balanced_records)),
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(train_dataset_full, train_subset.indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset_full, val_subset.indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = get_device()
    model = make_model(model_key, pretrained=use_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    progress = st.progress(0)
    status = st.empty()
    history: list[dict[str, float]] = []
    best_val_acc = 0.0
    best_state = None
    best_labels: list[int] = []
    best_preds: list[int] = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        labels_epoch: list[int] = []
        preds_epoch: list[int] = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(1)

                val_loss += loss.item()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                labels_epoch.extend(labels.cpu().tolist())
                preds_epoch.extend(preds.cpu().tolist())

        scheduler.step()

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, len(train_loader)),
            "val_loss": val_loss / max(1, len(val_loader)),
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_labels = labels_epoch
            best_preds = preds_epoch

        progress.progress((epoch + 1) / epochs)
        status.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}"
        )

    MODEL_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODEL_DIR / MODEL_OPTIONS[model_key]["checkpoint"].name
    if best_state is None:
        raise RuntimeError("Training selesai tanpa state model yang valid.")

    torch.save(best_state, checkpoint_path)
    load_model.clear()

    return {
        "checkpoint_path": checkpoint_path,
        "history": pd.DataFrame(history),
        "labels": best_labels,
        "preds": best_preds,
        "used_records": balanced_records,
    }


def render_header(records: list[ImageRecord]) -> None:
    st.title("Klasifikasi Citra Kebakaran Multi-Kelas")
    st.caption("MobileNetV3 / EfficientNet-B0 untuk citra kebakaran dan non-damage.")

    summary = summarize_records(records)
    total_images = int(summary["Jumlah Gambar"].sum()) if not summary.empty else 0
    cols = st.columns(5)
    cols[0].metric("Total Gambar", f"{total_images:,}")
    for index, class_name in enumerate(CLASS_NAMES, start=1):
        count = summary.loc[summary["Kelas"] == class_name, "Jumlah Gambar"].sum()
        cols[index].metric(class_name.replace("_", " "), f"{int(count):,}")

    if not records:
        st.info(
            "Dataset lokal tidak ditemukan. Ini aman untuk hosting prediksi: "
            "upload gambar di tab Prediksi selama file model `.pth` tersedia di folder `models`."
        )


def render_dataset_tab(records: list[ImageRecord]) -> None:
    st.subheader("Dataset Lokal")
    if not records:
        st.warning(
            f"Dataset tidak ditemukan di: {RAW_DATASET_DIR}. "
            "Tab ini hanya diperlukan untuk eksplorasi data atau training ulang."
        )
        return

    summary = summarize_records(records)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.bar_chart(summary.set_index("Kelas"))

    st.subheader("Contoh Gambar")
    columns = st.columns(len(CLASS_NAMES))
    rng = random.Random(42)
    for column, class_name in zip(columns, CLASS_NAMES):
        samples = [record for record in records if record.class_name == class_name]
        column.markdown(f"**{class_name.replace('_', ' ')}**")
        if not samples:
            column.info("Belum ada gambar.")
            continue
        sample = rng.choice(samples)
        column.image(str(sample.path), use_container_width=True)
        column.caption(sample.path.name)


def render_prediction_tab() -> None:
    st.subheader("Prediksi Gambar")
    if torch is None or timm is None:
        st.warning(
            "Dependency model belum lengkap. Jalankan "
            "`pip install -r Streamlit/requirements.txt` sebelum prediksi."
        )
        if TORCH_IMPORT_ERROR is not None:
            st.code(f"PyTorch error: {TORCH_IMPORT_ERROR}", language="text")
        if TIMM_IMPORT_ERROR is not None:
            st.code(f"timm error: {TIMM_IMPORT_ERROR}", language="text")
        return

    model_key = st.selectbox("Model", list(MODEL_OPTIONS.keys()), key="predict_model")
    checkpoint = find_checkpoint(model_key)

    if checkpoint is None:
        st.warning(
            "Checkpoint belum ditemukan. Latih model di tab Training, atau letakkan "
            f"file {MODEL_OPTIONS[model_key]['checkpoint'].name} di folder proyek."
        )
        return

    uploaded = st.file_uploader(
        "Upload gambar",
        type=sorted(ext.replace(".", "") for ext in IMAGE_EXTENSIONS),
    )
    if uploaded is None:
        st.info(f"Model siap digunakan dari: {checkpoint}")
        return

    image = Image.open(uploaded).convert("RGB")
    model, device = load_model(model_key, str(checkpoint))
    result = predict_image(image, model, device)
    top = result.iloc[0]

    left, right = st.columns([1, 1])
    left.image(image, caption=uploaded.name, use_container_width=True)
    right.metric("Prediksi", str(top["Kelas"]).replace("_", " "))
    right.metric("Confidence", f"{float(top['Probabilitas']):.2%}")
    right.dataframe(result, use_container_width=True, hide_index=True)


def render_training_tab(records: list[ImageRecord]) -> None:
    st.subheader("Training Model")
    if not records:
        st.warning(
            "Dataset belum tersedia, jadi training ulang tidak bisa dijalankan di sini. "
            "Untuk hosting gratis, ini normal karena app sebaiknya hanya menjalankan prediksi."
        )
        return
    if torch is None or timm is None:
        st.error(
            "Package torch, torchvision, atau timm belum terpasang. "
            "Jalankan `pip install -r Streamlit/requirements.txt`."
        )
        if TORCH_IMPORT_ERROR is not None:
            st.code(f"PyTorch error: {TORCH_IMPORT_ERROR}", language="text")
        if TIMM_IMPORT_ERROR is not None:
            st.code(f"timm error: {TIMM_IMPORT_ERROR}", language="text")
        return

    with st.form("training_form"):
        cols = st.columns(3)
        model_key = cols[0].selectbox("Arsitektur", list(MODEL_OPTIONS.keys()))
        epochs = cols[1].number_input("Epoch", min_value=1, max_value=50, value=5)
        batch_size = cols[2].number_input("Batch size", min_value=4, max_value=128, value=32, step=4)

        cols = st.columns(3)
        learning_rate = cols[0].number_input(
            "Learning rate",
            min_value=0.00001,
            max_value=0.1,
            value=0.001,
            format="%.5f",
        )
        max_non_damage = cols[1].number_input(
            "Maks data non-damage per kelas",
            min_value=50,
            max_value=5000,
            value=500,
            step=50,
        )
        seed = cols[2].number_input("Seed", min_value=1, max_value=9999, value=42)

        use_pretrained = st.checkbox(
            "Gunakan pretrained ImageNet",
            value=False,
            help="Aktifkan bila bobot pretrained sudah tersedia/cache atau internet aktif.",
        )
        submitted = st.form_submit_button("Mulai Training")

    if not submitted:
        st.info("Training akan menyimpan checkpoint terbaik ke folder `models`.")
        return

    with st.spinner("Training sedang berjalan..."):
        result = train_model(
            records=records,
            model_key=model_key,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            max_non_damage=int(max_non_damage),
            use_pretrained=use_pretrained,
            seed=int(seed),
        )

    st.success(f"Checkpoint tersimpan: {result['checkpoint_path']}")
    history = result["history"]
    st.line_chart(history.set_index("epoch")[["train_acc", "val_acc"]])
    st.line_chart(history.set_index("epoch")[["train_loss", "val_loss"]])

    labels = result["labels"]
    preds = result["preds"]
    report = classification_report(
        labels,
        preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    matrix = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    st.dataframe(
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES),
        use_container_width=True,
    )


def main() -> None:
    configure_page()
    records = scan_dataset(str(RAW_DATASET_DIR))
    render_header(records)

    dataset_tab, prediction_tab, training_tab = st.tabs(
        ["Dataset", "Prediksi", "Training"]
    )
    with dataset_tab:
        render_dataset_tab(records)
    with prediction_tab:
        render_prediction_tab()
    with training_tab:
        render_training_tab(records)


if __name__ == "__main__":
    main()
