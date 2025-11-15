"""Centralized model path configuration for NVIDIA medical models."""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
REFERENCE_DIR = BASE_DIR / "reference"

# Model paths with environment variable overrides
# All models are directly in ./models/ (flat structure)
MODEL_PATHS = {
    'nv-reason': os.getenv(
        'NV_REASON_PATH',
        str(MODELS_DIR / 'NV-Reason-CXR-3B')
    ),
    'nv-segment-ct': os.getenv(
        'NV_SEGMENT_CT_PATH',
        str(MODELS_DIR / 'NV-Segment-CT')
    ),
    'nv-segment-ctmr': os.getenv(
        'NV_SEGMENT_CTMR_PATH',
        str(MODELS_DIR / 'NV-Segment-CTMR')
    ),
    'nv-generate-ct': os.getenv(
        'NV_GENERATE_CT_PATH',
        str(MODELS_DIR / 'NV-Generate-CT')
    ),
    'nv-generate-mr': os.getenv(
        'NV_GENERATE_MR_PATH',
        str(MODELS_DIR / 'NV-Generate-MR')
    ),
}


def verify_model_paths(verbose=True):
    """
    Verify that configured model paths exist.

    Args:
        verbose: If True, print status for each model

    Returns:
        dict: Status of each model (True if exists, False otherwise)
    """
    status = {}

    for name, path in MODEL_PATHS.items():
        exists = Path(path).exists()
        status[name] = exists

        if verbose:
            if exists:
                print(f"✅ {name}: {path}")
            else:
                print(f"❌ {name}: Not found at {path}")

    return status


def get_missing_models():
    """
    Get list of models that are not downloaded.

    Returns:
        list: Names of missing models
    """
    status = verify_model_paths(verbose=False)
    return [name for name, exists in status.items() if not exists]


if __name__ == '__main__':
    # Run verification when executed directly
    print("Verifying model paths...\n")
    status = verify_model_paths(verbose=True)

    missing = [name for name, exists in status.items() if not exists]

    if missing:
        print(f"\n⚠️  Missing models: {', '.join(missing)}")
        print("\nRun the download commands from README.md to set up models.")
    else:
        print("\n✅ All models found!")
