IMAGE_SIZE = 224
DEFAULT_DATA_DIR = "data/domestic_garbage"
DEFAULT_OUTPUT_DIR = "outputs/expanded_balanced_efficientnet_b0"
DEFAULT_CHECKPOINT = f"{DEFAULT_OUTPUT_DIR}/best_model.pth"
DEFAULT_CLASS_NAMES = f"{DEFAULT_OUTPUT_DIR}/class_names.json"
DEFAULT_ENSEMBLE_MODELS = []
LEGACY_ENSEMBLE_MODELS = [
    (
        "outputs/domestic_four_class_retrain/best_model.pth",
        "outputs/domestic_four_class_retrain/class_names.json",
    ),
    (
        "outputs/domestic_four_class_efficientnet_b0/best_model.pth",
        "outputs/domestic_four_class_efficientnet_b0/class_names.json",
    ),
]
