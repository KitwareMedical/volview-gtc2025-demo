import asyncio
import os
import glob
import subprocess
import sys
import tempfile
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

import itk
from volview_server import VolViewApi, get_current_client_store
from volview_server.transformers import (
    convert_itk_to_vtkjs_image,
)

# --- Configuration ---
MAISI_BUNDLE_DIR = "bundles/"
MAISI_BUNDLE_NAME = "maisi_ct_generative"

# --- Global Setup ---
volview = VolViewApi()
process_pool = ProcessPoolExecutor(max_workers=2)


def do_maisi_generation(params: dict) -> Dict:
    """
    Performs MONAI bundle generation and returns the result as a vtk.js dictionary.

    This function is designed to be called via ProcessPoolExecutor. It handles:
    1. Downloading the MONAI bundle if not present.
    2. Running the MAISI bundle generation via a subprocess with given params.
    3. Finding the resulting image file from the output directory.
    4. Converting the result to a vtk.js-compatible dictionary for transport.

    Args:
        params: A dictionary of parameters to pass to the MONAI bundle.

    Returns:
        A dictionary representing the vtk.js image data of the generated CT.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        python_executable = sys.executable
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        bundle_root = os.path.join(MAISI_BUNDLE_DIR, MAISI_BUNDLE_NAME)

        # 1. Download the MONAI Bundle if necessary
        print("MONAI: Ensuring MAISI bundle is downloaded...")
        download_command = [
            python_executable, "-m", "monai.bundle", "download",
            "--name", MAISI_BUNDLE_NAME, "--bundle_dir", MAISI_BUNDLE_DIR,
        ]
        subprocess.run(download_command, check=True, capture_output=True, text=True)
        print("MONAI: Bundle is ready.")

        # 2. Execute generation. Note, the config must be relative to cwd, which is set below
        config_file_path = os.path.join("configs", "inference.json")
        
        print(f"MONAI: Running MAISI generation with params: {params}...")
        
        # Build the inference command dynamically from the received parameters
        inference_command = [
            python_executable, "-m", "monai.bundle", "run",
            "--config_file", config_file_path,
            f"--output_dir={output_dir}",
            "--num_output_samples=1",
        ]
        
        # Helper to format list arguments for the command line
        def format_list_for_cli(data: List[Any]) -> str:
            # json.dumps ensures proper quoting for strings inside the list
            return json.dumps(data)

        # Map frontend params to MONAI bundle override arguments
        if 'anatomy_list' in params and params['anatomy_list']:
            anatomy_arg = format_list_for_cli(params['anatomy_list'])
            inference_command.append(f"--anatomy_list={anatomy_arg}")
            
        if 'output_size' in params and len(params['output_size']) == 3:
            size_arg = format_list_for_cli(params['output_size'])
            inference_command.append(f"--output_size={size_arg}")
            
        if 'spacing' in params and len(params['spacing']) == 3:
            spacing_arg = format_list_for_cli(params['spacing'])
            inference_command.append(f"--spacing={spacing_arg}")

        print(f"MONAI: Executing command: {' '.join(inference_command)}")

        try:
            result = subprocess.run(
                inference_command,
                cwd=bundle_root,
                check=True,
                capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            # **MODIFICATION:** Instead of printing, format a detailed error message.
            error_message = (
                f"\n--- MONAI Subprocess Failed ---\n"
                f"Return Code: {e.returncode}\n"
                f"\n--- STDOUT ---\n{e.stdout}\n"
                f"\n--- STDERR ---\n{e.stderr}\n"
                f"-----------------------------"
            )
            # Raise a new exception that contains the formatted message.
            # This will be sent back to the main process.
            raise Exception(error_message)

        print("MONAI STDOUT:", result.stdout)
        if result.stderr:
            print("MONAI STDERR:", result.stderr)

        # 3. Find and read the generated result
        # Be more specific to get the image, not a mask file
        search_pattern = os.path.join(output_dir, "*image.nii.gz")
        results = glob.glob(search_pattern)
        
        if not results:
            # Fallback to any .nii.gz if specific search fails
            search_pattern_fallback = os.path.join(output_dir, "*.nii.gz")
            results = glob.glob(search_pattern_fallback)

        if not results:
            raise FileNotFoundError(
                "MONAI generation finished but the expected output file "
                f"was not found in {output_dir}"
            )

        result_path = results[0] # Get the first result
        print(f"MONAI: Found generated image at {result_path}")
        itk_image_result = itk.imread(result_path)

        # 4. Convert the ITK result to a vtk.js dictionary and return it
        print("MONAI: Converting ITK image to VTK.js format...")
        vtkjs_data = convert_itk_to_vtkjs_image(itk_image_result)

        print("MONAI: Generation complete. Returning generated CT object.")
        return vtkjs_data


async def run_monai_generation_process(params: dict) -> Dict:
    """
    Asynchronously runs the MONAI generation function in the process pool.
    """
    loop = asyncio.get_event_loop()
    generation_object = await loop.run_in_executor(
        process_pool, do_maisi_generation, params
    )
    return generation_object


@volview.expose("generateWithMAISI")
async def run_maisi_generation(generation_id: str, params: dict):
    """
    Exposes MONAI MAISI generation to the VolView client.

    Takes a unique generation ID and parameters from the client, runs the
    generation process, and sends the resulting vtk.js object back to the client.

    Args:
        generation_id: The unique ID for this generation task.
        params: A dictionary of parameters for the model.
    """
    print(f"Received MONAI generation request with ID: {generation_id}")
    
    generation_result_obj = await run_monai_generation_process(params)

    maisi_store = get_current_client_store("maisi")
    await maisi_store.setMAISIResult(generation_id, generation_result_obj)

    print("Successfully created CT. Sending object back to client.")
    return 0
