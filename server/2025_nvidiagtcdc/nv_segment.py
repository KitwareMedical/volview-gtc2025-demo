import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import itk
from volview_server import VolViewApi, get_current_client_store
from volview_server.transformers import (
    convert_itk_to_vtkjs_image,
    convert_vtkjs_to_itk_image,
)

# Import centralized model configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.model_paths import MODEL_PATHS

# --- Configuration ---

# Model registry with support for both CT and CTMR variants
SEGMENT_MODELS = {
    'NV-Segment-CT': {
        'bundle_path': MODEL_PATHS['nv-segment-ct'],
        'num_classes': 132,
        'modality': 'CT_BODY',
        'description': 'CT-only (132 classes), best for tumor segmentation',
    },
    'NV-Segment-CTMR': {
        'bundle_path': MODEL_PATHS['nv-segment-ctmr'],
        'num_classes': 345,
        'modality': ['CT_BODY', 'MRI_BODY', 'MRI_BRAIN'],
        'description': 'CT+MR (345 classes), includes brain structures',
    },
}

# --- Global Setup ---

volview = VolViewApi()
# It's crucial to run blocking, CPU-intensive tasks in a separate process
# to avoid stalling the async event loop.
process_pool = ProcessPoolExecutor(max_workers=2)


def _execute_vista3d_inference_in_process(
    vtkjs_image_dict: dict,
    label_prompt: List[int],
    bundle_path: str,
    modality: str = None
) -> Dict:
    """
    Runs VISTA-3D bundle inference in a separate process.

    This function is designed to be called via ProcessPoolExecutor. It handles
    the entire pipeline:
    1. Converts the vtk.js dictionary back into an ITK image.
    2. Saves the ITK image to a temporary NRRD file.
    3. Runs the MONAI bundle inference via a subprocess.
    4. Reads the resulting segmentation file from disk.
    5. Converts the resulting ITK image to a vtk.js-compatible dictionary.

    Args:
        vtkjs_image_dict: A dict representing a vtk.js image. This plain
            dictionary format is used for stable inter-process communication.
        label_prompt: List of class indices to segment. Empty list segments all.
        bundle_path: Path to the MONAI bundle directory.
        modality: Modality key for CTMR bundle ('CT_BODY', 'MRI_BODY', 'MRI_BRAIN').
                  None for CT-only model (doesn't support modality parameter).

    Returns:
        A dictionary representing the vtk.js image data of the segmentation.
    """
    # 1. Verify bundle exists
    bundle_root = Path(bundle_path)
    if not bundle_root.exists():
        raise FileNotFoundError(
            f"Bundle not found at {bundle_path}. "
            "Please run model download commands from README.md"
        )

    config_file = bundle_root / 'configs' / 'inference.json'
    if not config_file.exists():
        raise FileNotFoundError(
            f"Bundle config not found: {config_file}. "
            "Bundle may be incomplete or corrupted."
        )

    # 2. Convert incoming image data to an ITK image object
    input_itk_image = convert_vtkjs_to_itk_image(vtkjs_image_dict)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 3. Save the ITK image to a temporary file
        input_filename = "input_image.nrrd"
        tmp_image_path = os.path.join(tmpdir, input_filename)
        itk.imwrite(input_itk_image, tmp_image_path)
        abs_image_path = os.path.abspath(tmp_image_path)

        python_executable = sys.executable

        # 4. Clean up old eval directory
        eval_dir = bundle_root / "eval"
        if eval_dir.exists():
            shutil.rmtree(eval_dir)

        # 5. Build input_dict with optional label_prompt and modality
        input_dict_parts = [f"'image':'{abs_image_path}'"]

        if modality:
            input_dict_parts.append(f"'modality':'{modality}'")

        if label_prompt and len(label_prompt) > 0:
            input_dict_parts.append(f"'label_prompt':{label_prompt}")

        # Single log line with all relevant info
        modality_str = f" (modality={modality})" if modality else ""
        classes_str = f" classes={label_prompt}" if label_prompt else " all classes"
        print(f"NV-Segment: Running inference{modality_str},{classes_str}")

        input_dict = f"{{{','.join(input_dict_parts)}}}"

        # 6. Execute inference
        inference_command = [
            python_executable, "-m", "monai.bundle", "run",
            "--config_file", "configs/inference.json",
            "--input_dict", input_dict,
        ]

        try:
            result = subprocess.run(
                inference_command, cwd=str(bundle_root), check=True,
                capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Inference failed (exit {e.returncode})")
            print(f"STDERR:\n{e.stderr}")
            raise

        # 7. Read and convert segmentation result
        input_name_no_ext = os.path.splitext(input_filename)[0]
        result_path = os.path.join(
            bundle_root, "eval", input_name_no_ext, f"{input_name_no_ext}_trans.nii.gz"
        )

        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Output not found at {result_path}")

        itk_image_result = itk.imread(result_path)
        return convert_itk_to_vtkjs_image(itk_image_result)


async def run_vista3d_inference_async(
    itk_image: itk.Image,
    label_prompt: List[int],
    bundle_path: str,
    modality: str = None
) -> Dict:
    """
    Asynchronously runs the NV-Segment inference in the process pool.

    Args:
        itk_image: The ITK image object to segment.
        label_prompt: List of class indices to segment.
        bundle_path: Path to the MONAI bundle directory.
        modality: Modality key for CTMR bundle. None for CT-only model.

    Returns:
        The segmentation result as a vtk.js dictionary.
    """
    # Convert to vtk.js dict for stable serialization to the process pool
    vtkjs_image_dict = convert_itk_to_vtkjs_image(itk_image)

    loop = asyncio.get_event_loop()
    segmentation_vtkjs_dict = await loop.run_in_executor(
        process_pool,
        _execute_vista3d_inference_in_process,
        vtkjs_image_dict,
        label_prompt,
        bundle_path,
        modality
    )
    return segmentation_vtkjs_dict


@volview.expose("segmentWithNVSegment")
async def run_nv_segment_segmentation(
    img_id: str,
    label_prompt: List[int] = None,
    model_name: str = 'NV-Segment-CT',
    modality: str = 'CT'
):
    """
    Exposes NV-Segment (VISTA-3D) segmentation to the VolView client.

    Supports both CT-only (132 classes) and CTMR (345 classes) variants.

    Args:
        img_id: The ID of the image to segment.
        label_prompt: Optional list of class indices to segment.
                      Empty or None means segment all classes.
        model_name: Model variant to use ('NV-Segment-CT' or 'NV-Segment-CTMR').
        modality: Image modality ('CT', 'MR', or 'MR_BRAIN'). Only used for CTMR model.
    """
    if label_prompt is None:
        label_prompt = []

    # Validate model selection
    if model_name not in SEGMENT_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(SEGMENT_MODELS.keys())}"
        )

    model_config = SEGMENT_MODELS[model_name]
    bundle_path = model_config['bundle_path']

    # Only use modality for CTMR model (CT model doesn't support it)
    modality_key = None
    if model_name == 'NV-Segment-CTMR':
        modality_map = {
            'CT': 'CT_BODY',
            'MR': 'MRI_BODY',
            'MR_BRAIN': 'MRI_BRAIN',
        }
        modality_key = modality_map.get(modality, 'CT_BODY')

    print(f"Received {model_name} segmentation request for image ID: {img_id}")
    if modality_key:
        print(f"Modality: {modality} → {modality_key}")
    print(f"Class selection: {label_prompt if label_prompt else 'All classes'}")

    # Clear old segmentation results to free memory
    nv_segment_store = get_current_client_store("nv-segment")
    try:
        old_ids = await nv_segment_store.nvSegmentIds
        if old_ids:
            for old_id in old_ids:
                await nv_segment_store.removeNVSegmentResult(old_id)
    except Exception:
        pass  # Non-critical, continue if cleanup fails

    # Fetch and convert image
    # Disable auto-deserialization to get raw vtkjs format
    image_cache_store = get_current_client_store("image-cache", transform_args=False)
    vtkjs_image = await image_cache_store.getVtkImageData(img_id)
    if vtkjs_image is None:
        raise ValueError(f"No image found for ID: {img_id}")

    itk_image = convert_vtkjs_to_itk_image(vtkjs_image)

    # Run segmentation
    segmentation_vtkjs_dict = await run_vista3d_inference_async(
        itk_image, label_prompt, bundle_path, modality_key
    )

    await nv_segment_store.setNVSegmentResult(img_id, segmentation_vtkjs_dict)
    return 0
