import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

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


def do_monai_inference(serialized_img: dict) -> Dict:
    """
    Performs MONAI bundle inference and returns the result as a vtk.js dictionary.

    This function is designed to be called via ProcessPoolExecutor. It handles
    the entire pipeline:
    1. Deserializes the input image.
    2. Saves it to a temporary NRRD file.
    3. Downloads the MONAI bundle if not present.
    4. Runs the MONAI bundle inference via a subprocess.
    5. Reads the resulting segmentation file from disk.
    6. Converts the result to a vtk.js-compatible dictionary for transport.

    Args:
        serialized_img: A vtkjs-serialized image dictionary.

    Returns:
        A dictionary representing the vtk.js image data of the segmentation.
    """
    # 1. Convert incoming image data to an ITK image object
    input_itk_image = convert_vtkjs_to_itk_image(serialized_img)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2. Save the ITK image to a temporary file
        input_filename = "input_image.nrrd"
        tmp_image_path = os.path.join(tmpdir, input_filename)
        itk.imwrite(input_itk_image, tmp_image_path)
        abs_image_path = os.path.abspath(tmp_image_path)

        python_executable = sys.executable

        # 3. Download the MONAI Bundle if necessary
        print("MONAI: Ensuring Vista3D bundle is downloaded...")
        download_command = [
            python_executable, "-m", "monai.bundle", "download",
            VISTA3D_BUNDLE_NAME, "--bundle_dir", VISTA3D_BUNDLE_DIR,
        ]
        subprocess.run(download_command, check=True, capture_output=True, text=True)
        print("MONAI: Bundle is ready.")

        # 4. Execute inference
        bundle_root = os.path.join(VISTA3D_BUNDLE_DIR, VISTA3D_BUNDLE_NAME)
        eval_dir = os.path.join(bundle_root, "eval")
        if os.path.exists(eval_dir):
            shutil.rmtree(eval_dir)

        print(f"MONAI: Running inference on {abs_image_path}...")
        inference_command = [
            python_executable, "-m", "monai.bundle", "run",
            "--config_file", "configs/inference.json",
            "--input_dict", f"{{'image':'{abs_image_path}'}}",
        ]
        result = subprocess.run(
            inference_command, cwd=bundle_root, check=True,
            capture_output=True, text=True
        )
        print("MONAI STDOUT:", result.stdout)
        if result.stderr:
            print("MONAI STDERR:", result.stderr)

        # 5. Find and read the segmentation result
        input_name_no_ext = os.path.splitext(input_filename)[0]
        output_filename = f"{input_name_no_ext}_trans.nii.gz"
        result_path = os.path.join(
            bundle_root, "eval", input_name_no_ext, output_filename
        )
        if not os.path.exists(result_path):
            raise FileNotFoundError(
                "MONAI inference finished but the expected output file "
                f"was not found at {result_path}"
            )
        itk_image_result = itk.imread(result_path)

        # 6. Convert the ITK result to a vtk.js dictionary and return it
        print("MONAI: Converting ITK image to VTK.js format...")
        vtkjs_data = convert_itk_to_vtkjs_image(itk_image_result)

        print("MONAI: Inference complete. Returning segmentation object.")
        return vtkjs_data


async def run_monai_inference_process(img) -> Dict:
    """
    Asynchronously runs the MONAI inference function in the process pool.
    """
    serialized_img = convert_itk_to_vtkjs_image(img)
    loop = asyncio.get_event_loop()
    segmentation_object = await loop.run_in_executor(
        process_pool, do_monai_inference, serialized_img
    )
    return segmentation_object


@volview.expose("segmentWithMONAI")
async def run_vista3d_segmentation(img_id: str):
    """
    Exposes MONAI Vista3D segmentation to the VolView client.

    Takes an image ID from the client, runs inference, and sends the
    resulting vtk.js object back to the client.

    Args:
        img_id: The ID of the image to segment.
    """
    print(f"Received MONAI segmentation request for image ID: {img_id}")
    image_cache_store = get_current_client_store("image-cache")

    img = await image_cache_store.getVtkImageData(img_id)
    if img is None:
        raise ValueError(f"No image found for ID: {img_id}")

    segmentation_result_obj = await run_monai_inference_process(img)

    vista3d_store = get_current_client_store("vista3d")
    await vista3d_store.setVista3dResult(img_id, segmentation_result_obj)

    print("Successfully created segmentation. Sending object back to client.")
    return 0
