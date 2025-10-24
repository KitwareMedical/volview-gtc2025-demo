import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import itk
from volview_server import VolViewApi, get_current_client_store
from volview_server.transformers import (
    convert_itk_to_vtkjs_image,
    convert_vtkjs_to_itk_image,
)

# --- Configuration ---

# This should point to the directory where you downloaded the MONAI bundles.
# Example: "bundles/"
VISTA3D_BUNDLE_DIR = "bundles/"
VISTA3D_BUNDLE_NAME = "vista3d"

# --- Global Setup ---

volview = VolViewApi()
# It's crucial to run blocking, CPU-intensive tasks in a separate process
# to avoid stalling the async event loop.
process_pool = ProcessPoolExecutor(max_workers=2)


def _execute_vista3d_inference_in_process(
    vtkjs_image_dict: dict, label_prompt: List[int]
) -> Dict:
    """
    Runs VISTA-3D bundle inference in a separate process.

    This function is designed to be called via ProcessPoolExecutor. It handles
    the entire pipeline:
    1. Converts the vtk.js dictionary back into an ITK image.
    2. Saves the ITK image to a temporary NRRD file.
    3. Downloads the MONAI bundle if not present.
    4. Runs the 'vista3d' bundle inference via a subprocess.
    5. Reads the resulting segmentation file from disk.
    6. Converts the resulting ITK image to a vtk.js-compatible dictionary.

    Args:
        vtkjs_image_dict: A dict representing a vtk.js image. This plain
            dictionary format is used for stable inter-process communication.
        label_prompt: List of class indices to segment. Empty list segments all.

    Returns:
        A dictionary representing the vtk.js image data of the segmentation.
    """
    # 1. Convert incoming image data to an ITK image object
    input_itk_image = convert_vtkjs_to_itk_image(vtkjs_image_dict)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2. Save the ITK image to a temporary file
        input_filename = "input_image.nrrd"
        tmp_image_path = os.path.join(tmpdir, input_filename)
        itk.imwrite(input_itk_image, tmp_image_path)
        abs_image_path = os.path.abspath(tmp_image_path)

        python_executable = sys.executable

        # 3. Download the MONAI Bundle if necessary
        print("VISTA-3D: Ensuring bundle is downloaded...")
        download_command = [
            python_executable, "-m", "monai.bundle", "download",
            VISTA3D_BUNDLE_NAME, "--bundle_dir", VISTA3D_BUNDLE_DIR,
        ]
        subprocess.run(download_command, check=True, capture_output=True, text=True)
        print("VISTA-3D: Bundle is ready.")

        # 4. Execute inference
        bundle_root = os.path.join(VISTA3D_BUNDLE_DIR, VISTA3D_BUNDLE_NAME)
        eval_dir = os.path.join(bundle_root, "eval")
        if os.path.exists(eval_dir):
            shutil.rmtree(eval_dir)

        print(f"VISTA-3D: Running inference on {abs_image_path}...")

        # Build input_dict with optional label_prompt
        if label_prompt and len(label_prompt) > 0:
            input_dict = f"{{'image':'{abs_image_path}','label_prompt':{label_prompt}}}"
            print(f"VISTA-3D: Segmenting specific classes: {label_prompt}")
        else:
            input_dict = f"{{'image':'{abs_image_path}'}}"
            print("VISTA-3D: Segmenting all available classes")

        inference_command = [
            python_executable, "-m", "monai.bundle", "run",
            "--config_file", "configs/inference.json",
            "--input_dict", input_dict,
        ]
        result = subprocess.run(
            inference_command, cwd=bundle_root, check=True,
            capture_output=True, text=True
        )
        print("VISTA-3D STDOUT:", result.stdout)
        if result.stderr:
            print("VISTA-3D STDERR:", result.stderr)

        # 5. Find and read the segmentation result
        input_name_no_ext = os.path.splitext(input_filename)[0]
        output_filename = f"{input_name_no_ext}_trans.nii.gz"
        result_path = os.path.join(
            bundle_root, "eval", input_name_no_ext, output_filename
        )
        if not os.path.exists(result_path):
            raise FileNotFoundError(
                "VISTA-3D inference finished but the expected output file "
                f"was not found at {result_path}"
            )
        itk_image_result = itk.imread(result_path)

        # 6. Convert the ITK result to a vtk.js dictionary and return it
        print("VISTA-3D: Converting result to VTK.js format...")
        segmentation_vtkjs_dict = convert_itk_to_vtkjs_image(itk_image_result)

        print("VISTA-3D: Inference complete. Returning segmentation.")
        return segmentation_vtkjs_dict


async def run_vista3d_inference_async(
    itk_image: itk.Image, label_prompt: List[int]
) -> Dict:
    """
    Asynchronously runs the VISTA-3D inference in the process pool.

    Args:
        itk_image: The ITK image object to segment.
        label_prompt: List of class indices to segment.

    Returns:
        The segmentation result as a vtk.js dictionary.
    """
    # Convert to vtk.js dict for stable serialization to the process pool
    vtkjs_image_dict = convert_itk_to_vtkjs_image(itk_image)

    loop = asyncio.get_event_loop()
    segmentation_vtkjs_dict = await loop.run_in_executor(
        process_pool, _execute_vista3d_inference_in_process, vtkjs_image_dict, label_prompt
    )
    return segmentation_vtkjs_dict


@volview.expose("segmentWithNVSegmentCT")
async def run_nv_segment_ct_segmentation(img_id: str, label_prompt: List[int] = None):
    """
    Exposes MONAI VISTA-3D (NV-Segment-CT) segmentation to the VolView client.

    Takes an image ID from the client, runs inference, and sends the
    resulting segmentation (as a vtk.js object) back to the client.

    Args:
        img_id: The ID of the image to segment.
        label_prompt: Optional list of class indices to segment.
                      Empty or None means segment all classes.
    """
    if label_prompt is None:
        label_prompt = []

    print(f"Received NV-Segment-CT segmentation request for image ID: {img_id}")
    print(f"Class selection: {label_prompt if label_prompt else 'All classes'}")

    image_cache_store = get_current_client_store("image-cache")

    itk_image = await image_cache_store.getVtkImageData(img_id)
    if itk_image is None:
        raise ValueError(f"No image found for ID: {img_id}")

    segmentation_vtkjs_dict = await run_vista3d_inference_async(itk_image, label_prompt)

    nv_segment_store = get_current_client_store("nv-segment")
    await nv_segment_store.setNVSegmentResult(img_id, segmentation_vtkjs_dict)

    print("Successfully created segmentation. Sending object back to client.")
    return 0
