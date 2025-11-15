import asyncio
import os
import sys
import tempfile
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Any

import itk
from volview_server import VolViewApi, get_current_client_store
from volview_server.transformers import (
    convert_itk_to_vtkjs_image,
)

# Import centralized model configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.model_paths import MODEL_PATHS

# --- Configuration ---

# Model registry - CT and MR use separate bundles with separate datasets
GENERATE_MODELS = {
    'NV-Generate-CT': {
        'bundle_path': MODEL_PATHS['nv-generate-ct'],
        'version': 'rflow-ct',
        'modality': 'CT',
        'description': 'Synthetic CT generation (RFLOW, 30 steps)',
    },
    'NV-Generate-MR': {
        'bundle_path': MODEL_PATHS['nv-generate-mr'],
        'version': 'rflow-mr',
        'modality': 'MR',
        'description': 'Synthetic MR generation (RFLOW, 30 steps)',
    },
}

# --- Startup Validation ---

def validate_bundle_installation(model_name: str, bundle_path: str, version: str) -> None:
    """
    Validate that a NV-Generate bundle is properly installed.
    Fails fast with clear error message if bundle is incomplete.

    Args:
        model_name: Display name of the model (e.g., 'NV-Generate-CT')
        bundle_path: Path to the bundle directory
        version: Model version (e.g., 'rflow-ct')

    Raises:
        FileNotFoundError: If bundle or required files are missing
    """
    bundle_root = Path(bundle_path)

    if not bundle_root.exists():
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"❌ {model_name} NOT INSTALLED\n"
            f"{'='*70}\n"
            f"Bundle directory not found: {bundle_path}\n\n"
            f"Please follow the installation instructions in README.md:\n"
            f"1. Clone GitHub repo: git clone https://github.com/NVIDIA-Medtech/NV-Generate-CTMR.git\n"
            f"2. Copy configs/scripts to {bundle_path}/\n"
            f"3. Download models: huggingface-cli download nvidia/{model_name}\n"
            f"{'='*70}"
        )

    # Check required files for diff_model_infer workflow
    required_files = [
        'scripts/diff_model_infer.py',
        f'configs/config_maisi_diff_model_{version}.json',
        f'configs/environment_maisi_diff_model_{version}.json',
        'configs/config_network_rflow.json',
    ]

    missing_files = []
    for file_path in required_files:
        if not (bundle_root / file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"❌ {model_name} INCOMPLETE INSTALLATION\n"
            f"{'='*70}\n"
            f"Bundle found at: {bundle_path}\n"
            f"Missing required files:\n" +
            '\n'.join(f"  - {f}" for f in missing_files) + "\n\n"
            f"Please complete the installation from README.md\n"
            f"{'='*70}"
        )

    print(f"✅ {model_name} validated at {bundle_path}")


def validate_all_bundles():
    """Validate all NV-Generate bundles on startup. Fails fast if any are missing."""
    print("\n" + "="*70)
    print("VALIDATING NV-GENERATE MODELS")
    print("="*70)

    for model_name, config in GENERATE_MODELS.items():
        validate_bundle_installation(
            model_name,
            config['bundle_path'],
            config['version']
        )

    print("="*70)
    print("✅ All NV-Generate models validated successfully!")
    print("="*70 + "\n")


# Validate bundles on module import (server startup)
validate_all_bundles()

# --- Global Setup ---
volview = VolViewApi()
process_pool = ProcessPoolExecutor(max_workers=2)


def do_maisi_generation(
    params: dict,
    bundle_path: str,
    version: str
) -> Dict:
    """
    Performs NV-Generate generation using simplified diff_model_infer workflow.

    This function is designed to be called via ProcessPoolExecutor. It handles:
    1. Verifying the bundle exists (process-level safety check).
    2. Creating a temporary inference config with user params.
    3. Running scripts.diff_model_infer (simpler than full inference.py).
    4. Finding and converting the resulting image file.

    Args:
        params: A dictionary of parameters (output_size, spacing, etc.).
        bundle_path: Path to the NV-Generate bundle directory.
        version: Model version ('rflow-ct' or 'rflow-mr').

    Returns:
        A dictionary representing the vtk.js image data of the generated CT/MR.
    """
    # Process-level safety check (ProcessPoolExecutor runs in separate process)
    bundle_root = Path(bundle_path)
    if not bundle_root.exists():
        raise FileNotFoundError(
            f"Bundle not found at {bundle_path}. "
            "This should have been caught at server startup. Please check installation."
        )

    scripts_dir = bundle_root / 'scripts'
    if not scripts_dir.exists():
        raise FileNotFoundError(
            f"Scripts directory not found: {scripts_dir}. "
            "Please ensure bundle installation is complete (see README.md)"
        )

    # Add bundle to Python path to import diff_model_infer
    sys.path.insert(0, str(bundle_root))
    try:
        from scripts.diff_model_infer import diff_model_infer
    finally:
        sys.path.remove(str(bundle_root))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Modality mapping: CT=1-7, MR=8-20 (from modality_mapping.json)
        modality_value = 1 if version == 'rflow-ct' else 9  # CT body or MR body (9 = MR brain)

        # CFG guidance scale: CT uses 0 (no guidance), MR uses 15 (strong guidance for better quality)
        # MR images have higher variability and benefit significantly from classifier-free guidance
        default_cfg_scale = 0 if version == 'rflow-ct' else 15

        # Create temporary model config with user parameters
        model_config = {
            "diffusion_unet_train": {
                "batch_size": 1,
                "cache_rate": 0,
                "lr": 0.00001,
                "n_epochs": 1000
            },
            "diffusion_unet_inference": {
                "dim": params.get('output_size', [256, 256, 128]),
                "spacing": params.get('spacing', [1.5, 1.5, 1.5]),
                "top_region_index": [0, 1, 0, 0],
                "bottom_region_index": [0, 0, 1, 0],
                "random_seed": params.get('random_seed', 0),
                "num_inference_steps": 30,
                "modality": modality_value,
                "cfg_guidance_scale": params.get('cfg_guidance_scale', default_cfg_scale)
            }
        }

        # Write temporary model config
        temp_model_config_path = os.path.join(tmpdir, "config_maisi_diff_model_temp.json")
        with open(temp_model_config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        # Create temporary environment config with custom output dir
        temp_env_config_path = os.path.join(tmpdir, "environment_temp.json")
        env_config = {
            "data_base_dir": "./data",
            "embedding_base_dir": "./embeddings",
            "json_data_list": "./dataset.json",
            "model_dir": "./models",
            "model_filename": f"diff_unet_3d_{version}.pt",
            "output_dir": output_dir,
            "output_prefix": "generated",
            "trained_autoencoder_path": "models/autoencoder_v2.pt" if version == 'rflow-mr' else "models/autoencoder_v1.pt",
            "existing_ckpt_filepath": f"./models/diff_unet_3d_{version}.pt",
            "modality_mapping_path": "./configs/modality_mapping.json"
        }
        with open(temp_env_config_path, 'w') as f:
            json.dump(env_config, f, indent=2)

        print(f"NV-Generate: Using bundle at {bundle_root}")
        print(f"NV-Generate: Version={version}, Params={params}")

        # Run diff_model_infer
        # Change to bundle directory for relative paths to work
        original_cwd = os.getcwd()
        try:
            os.chdir(str(bundle_root))

            output_paths = diff_model_infer(
                env_config_path=temp_env_config_path,
                model_config_path=temp_model_config_path,
                model_def_path="./configs/config_network_rflow.json",
                num_gpus=1
            )

            if not output_paths or not output_paths[0]:
                raise FileNotFoundError(
                    "NV-Generate: Generation finished but no output file was returned"
                )

            result_path = output_paths[0]
            print(f"NV-Generate: Found generated image at {result_path}")

        finally:
            os.chdir(original_cwd)

        # Read and convert the generated result
        itk_image_result = itk.imread(result_path)

        print("NV-Generate: Converting ITK image to VTK.js format...")
        vtkjs_data = convert_itk_to_vtkjs_image(itk_image_result)

        print("NV-Generate: Generation complete. Returning generated object.")
        return vtkjs_data


async def run_monai_generation_process(
    params: dict,
    bundle_path: str,
    version: str
) -> Dict:
    """
    Asynchronously runs the NV-Generate function in the process pool.

    Args:
        params: Generation parameters.
        bundle_path: Path to the NV-Generate bundle.
        version: Model version ('rflow-ct' or 'rflow-mr').

    Returns:
        The generated image as a vtk.js dictionary.
    """
    loop = asyncio.get_event_loop()
    generation_object = await loop.run_in_executor(
        process_pool, do_maisi_generation, params, bundle_path, version
    )
    return generation_object


@volview.expose("generateWithNVGenerate")
async def run_nv_generate(
    generation_id: str,
    params: dict,
    model_name: str = 'NV-Generate-CT'
):
    """
    Exposes NV-Generate (MAISI) generation to the VolView client.

    Supports both CT and MR synthetic image generation.

    Args:
        generation_id: The unique ID for this generation task.
        params: A dictionary of parameters for the model.
        model_name: Model variant to use ('NV-Generate-CT' or 'NV-Generate-MR').
    """
    # Validate model selection
    if model_name not in GENERATE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(GENERATE_MODELS.keys())}"
        )

    model_config = GENERATE_MODELS[model_name]
    bundle_path = model_config['bundle_path']
    version = model_config['version']
    modality = model_config['modality']

    print(f"Received {model_name} generation request with ID: {generation_id}")
    print(f"Modality: {modality}, Version: {version}")
    print(f"Parameters: {params}")

    generation_result_obj = await run_monai_generation_process(
        params, bundle_path, version
    )

    nv_generate_store = get_current_client_store("nv-generate")
    await nv_generate_store.setNVGenerateResult(generation_id, generation_result_obj)

    print(f"Successfully created {modality}. Sending object back to client.")
    return 0


@volview.expose("generateWithMAISI")
async def run_maisi_generation(generation_id: str, params: dict):
    """
    Backwards compatibility wrapper for generateWithMAISI.
    Calls the new unified endpoint with CT defaults.
    """
    return await run_nv_generate(
        generation_id, params, model_name='NV-Generate-CT'
    )
