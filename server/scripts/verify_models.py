#!/usr/bin/env python3
"""
Model verification script for NVIDIA medical AI models.

This script verifies that all required models are properly downloaded and
configured. It checks for model files, configuration files, and provides
detailed error messages for troubleshooting.
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.model_paths import MODEL_PATHS, BASE_DIR


def check_file_size(path: Path) -> str:
    """Get human-readable file/directory size."""
    if not path.exists():
        return "N/A"

    if path.is_file():
        size_bytes = path.stat().st_size
    else:
        # For directories, sum all files
        size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def verify_nv_reason(model_path: str) -> tuple[bool, list[str]]:
    """Verify NV-Reason-CXR-3B model installation."""
    issues = []
    path = Path(model_path)

    if not path.exists():
        issues.append(f"Model directory not found: {model_path}")
        return False, issues

    # Check for essential model files
    required_files = [
        'config.json',
        'model.safetensors.index.json',
        'tokenizer_config.json',
        'preprocessor_config.json',
    ]

    for file in required_files:
        file_path = path / file
        if not file_path.exists():
            issues.append(f"Missing required file: {file}")

    # Check for model weights (at least one safetensors file)
    safetensors_files = list(path.glob('model*.safetensors'))
    if not safetensors_files:
        issues.append("No model weight files (*.safetensors) found")

    return len(issues) == 0, issues


def verify_nv_segment(model_path: str, model_name: str = None) -> tuple[bool, list[str]]:
    """Verify NV-Segment (VISTA-3D) bundle installation."""
    issues = []
    path = Path(model_path)

    if not path.exists():
        issues.append(f"Bundle directory not found: {model_path}")
        return False, issues

    # Check MONAI bundle structure
    configs_dir = path / 'configs'
    if not configs_dir.exists():
        issues.append("Missing 'configs/' directory")
    else:
        # Check for inference config
        inference_config = configs_dir / 'inference.json'
        if not inference_config.exists():
            issues.append("Missing configs/inference.json")

    # Check for model weights - different models expect different filenames
    models_dir = path / 'models'
    if not models_dir.exists():
        issues.append("Missing 'models/' directory")
    else:
        # CT model expects model_mrct.pt, CTMR expects model.pt
        if model_name == 'nv-segment-ct':
            expected_file = models_dir / 'model_mrct.pt'
            if not expected_file.exists():
                issues.append("Missing models/model_mrct.pt (expected by CT config)")
        elif model_name == 'nv-segment-ctmr':
            expected_file = models_dir / 'model.pt'
            if not expected_file.exists():
                issues.append("Missing models/model.pt (expected by CTMR config)")
        else:
            # Fallback: check for any .pt or .pth file
            model_files = list(models_dir.glob('*.pt')) + list(models_dir.glob('*.pth'))
            if not model_files:
                issues.append("No model weight files (*.pt, *.pth) found in models/")

    return len(issues) == 0, issues


def verify_nv_generate_ct(model_path: str) -> tuple[bool, list[str]]:
    """Verify NV-Generate-CT bundle installation (diff_model_infer workflow)."""
    issues = []
    path = Path(model_path)

    if not path.exists():
        issues.append(f"Bundle directory not found: {model_path}")
        return False, issues

    # Check for scripts directory
    scripts_dir = path / 'scripts'
    if not scripts_dir.exists():
        issues.append("Missing 'scripts/' directory")
    else:
        # Check for diff_model_infer script
        diff_infer_script = scripts_dir / 'diff_model_infer.py'
        if not diff_infer_script.exists():
            issues.append("Missing scripts/diff_model_infer.py")

    # Check MONAI bundle structure
    configs_dir = path / 'configs'
    if not configs_dir.exists():
        issues.append("Missing 'configs/' directory")
    else:
        # Check for CT diff_model configs
        ct_env = configs_dir / 'environment_maisi_diff_model_rflow-ct.json'
        if not ct_env.exists():
            issues.append("Missing configs/environment_maisi_diff_model_rflow-ct.json")

        ct_model_config = configs_dir / 'config_maisi_diff_model_rflow-ct.json'
        if not ct_model_config.exists():
            issues.append("Missing configs/config_maisi_diff_model_rflow-ct.json")

        # Check for network config
        network_config = configs_dir / 'config_network_rflow.json'
        if not network_config.exists():
            issues.append("Missing configs/config_network_rflow.json")

    # Check for model weights directory
    models_dir = path / 'models'
    if not models_dir.exists():
        issues.append("Missing 'models/' directory")
    else:
        # Check for CT-specific model files (only autoencoder + diffusion for diff_model_infer)
        required_models = [
            'autoencoder_v1.pt',
            'diff_unet_3d_rflow-ct.pt',
        ]
        for model_file in required_models:
            if not (models_dir / model_file).exists():
                issues.append(f"Missing models/{model_file}")

    return len(issues) == 0, issues


def verify_nv_generate_mr(model_path: str) -> tuple[bool, list[str]]:
    """Verify NV-Generate-MR bundle installation (diff_model_infer workflow)."""
    issues = []
    path = Path(model_path)

    if not path.exists():
        issues.append(f"Bundle directory not found: {model_path}")
        return False, issues

    # Check for scripts directory
    scripts_dir = path / 'scripts'
    if not scripts_dir.exists():
        issues.append("Missing 'scripts/' directory")
    else:
        # Check for diff_model_infer script
        diff_infer_script = scripts_dir / 'diff_model_infer.py'
        if not diff_infer_script.exists():
            issues.append("Missing scripts/diff_model_infer.py")

    # Check MONAI bundle structure
    configs_dir = path / 'configs'
    if not configs_dir.exists():
        issues.append("Missing 'configs/' directory")
    else:
        # Check for MR diff_model configs
        mr_env = configs_dir / 'environment_maisi_diff_model_rflow-mr.json'
        if not mr_env.exists():
            issues.append("Missing configs/environment_maisi_diff_model_rflow-mr.json")

        mr_model_config = configs_dir / 'config_maisi_diff_model_rflow-mr.json'
        if not mr_model_config.exists():
            issues.append("Missing configs/config_maisi_diff_model_rflow-mr.json")

        # Check for network config
        network_config = configs_dir / 'config_network_rflow.json'
        if not network_config.exists():
            issues.append("Missing configs/config_network_rflow.json")

    # Check for model weights directory
    models_dir = path / 'models'
    if not models_dir.exists():
        issues.append("Missing 'models/' directory")
    else:
        # Check for MR-specific model files (only autoencoder + diffusion for diff_model_infer)
        required_models = [
            'autoencoder_v2.pt',
            'diff_unet_3d_rflow-mr.pt',
        ]
        for model_file in required_models:
            if not (models_dir / model_file).exists():
                issues.append(f"Missing models/{model_file}")

    return len(issues) == 0, issues




def main():
    """Main verification function."""
    print("="*70)
    print("NVIDIA Medical AI Models - Installation Verification")
    print("="*70)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Models directory: {BASE_DIR / 'models'}\n")

    # Model configurations: display name and verification function
    models = {
        'nv-reason': ('NV-Reason-CXR-3B', verify_nv_reason),
        'nv-segment-ct': ('NV-Segment-CT (132 classes)', lambda p: verify_nv_segment(p, 'nv-segment-ct')),
        'nv-segment-ctmr': ('NV-Segment-CTMR (345+ classes)', lambda p: verify_nv_segment(p, 'nv-segment-ctmr')),
        'nv-generate-ct': ('NV-Generate-CT', verify_nv_generate_ct),
        'nv-generate-mr': ('NV-Generate-MR', verify_nv_generate_mr),
    }

    results = {}
    all_passed = True

    # Verify each model
    for model_key, (display_name, verify_func) in models.items():
        model_path = MODEL_PATHS[model_key]

        print(f"Checking {display_name}...")
        print(f"  Path: {model_path}")

        passed, issues = verify_func(model_path)
        results[model_key] = (passed, issues)

        if passed:
            size = check_file_size(Path(model_path))
            print(f"  Status: ✅ PASSED")
            print(f"  Size: {size}")
        else:
            print(f"  Status: ❌ FAILED")
            for issue in issues:
                print(f"    - {issue}")
            all_passed = False

        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    passed_count = sum(1 for passed, _ in results.values() if passed)
    total_count = len(results)

    print(f"\nModels verified: {passed_count}/{total_count}")

    if all_passed:
        print("\n✅ All models are properly installed and configured!")
        print("\nYou can now start the backend servers:")
        print("  cd server")
        print("  poetry run python -m volview_server -P 4014 -H 0.0.0.0 2025_nvidiagtcdc/nv_segment.py")
        print("  poetry run python -m volview_server -P 4015 -H 0.0.0.0 2025_nvidiagtcdc/chat.py")
        print("  poetry run python -m volview_server -P 4016 -H 0.0.0.0 2025_nvidiagtcdc/nv_generate.py")
        return 0
    else:
        print("\n❌ Some models are missing or incomplete.")
        print("\nTo fix missing models, run the download commands from README.md:")
        print("  - Section: 📦 Model Setup")
        print("\nOr run individual download commands as shown above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
