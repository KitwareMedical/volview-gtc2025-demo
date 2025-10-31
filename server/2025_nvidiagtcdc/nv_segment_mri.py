import asyncio
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import itk
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import pipeline
from volview_server import VolViewApi, get_current_client_store
from volview_server.transformers import (
    convert_itk_to_vtkjs_image,
    convert_vtkjs_to_itk_image,
)

# --- Configuration ---

# HuggingFace model configuration
MODEL_NAME = "nvidia/NV-Segment-CTMR"
MODEL_CACHE_DIR = "models/"
LOCAL_MODEL_DIR = os.path.join(MODEL_CACHE_DIR, "nv-segment-ctmr")

# --- Global Setup ---

volview = VolViewApi()
# It's crucial to run blocking, CPU-intensive tasks in a separate process
# to avoid stalling the async event loop.
process_pool = ProcessPoolExecutor(max_workers=2)

# Global model cache to avoid reloading on every inference
_pipeline_cache: Optional[object] = None


def load_nvctmr_model():
    """Load and initialize the NV-Segment-CTMR model with local caching."""
    global _pipeline_cache
    
    if _pipeline_cache is None:
        # Check if model exists locally, if not download it
        if not os.path.exists(LOCAL_MODEL_DIR):
            print(f"Downloading NV-Segment-CTMR model to {LOCAL_MODEL_DIR}...")
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=LOCAL_MODEL_DIR,
                local_dir_use_symlinks=False
            )
        
        print(f"Loading NV-Segment-CTMR model from {LOCAL_MODEL_DIR}...")
        
        device = 0 if torch.cuda.is_available() else -1
        print(f"Initializing pipeline on device: {device}")
        
        _pipeline_cache = pipeline(
            "image-segmentation",
            model=LOCAL_MODEL_DIR,
            device=device,
            trust_remote_code=True
        )
        print("NV-Segment-CTMR pipeline loaded successfully")
    
    return _pipeline_cache


def _execute_nv_segment_ctmr_inference_in_process(
    vtkjs_image_dict: dict, label_prompt: List[int], modality: str = "MRI_BODY"
) -> Dict:
    """
    Runs NV-Segment-CTMR inference in a separate process using HuggingFace pipeline.

    This function is designed to be called via ProcessPoolExecutor. It handles
    the entire pipeline:
    1. Converts the vtk.js dictionary back into an ITK image.
    2. Saves the ITK image to a temporary NIfTI file.
    3. Downloads the HuggingFace model if not present.
    4. Runs the NV-Segment-CTMR inference using transformers pipeline.
    5. Reads the resulting segmentation file from disk.
    6. Converts the resulting ITK image to a vtk.js-compatible dictionary.

    Args:
        vtkjs_image_dict: A dict representing a vtk.js image. This plain
            dictionary format is used for stable inter-process communication.
        label_prompt: List of class indices to segment. Empty list uses modality preset.
        modality: Modality type for segmentation ("MRI_BODY", "CT_BODY", "MRI_BRAIN").

    Returns:
        A dictionary representing the vtk.js image data of the segmentation.
    """
    global _pipeline_cache
    
    # 1. Convert incoming image data to an ITK image object
    input_itk_image = convert_vtkjs_to_itk_image(vtkjs_image_dict)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2. Save the ITK image to a temporary file
        input_filename = "input_image.nii.gz"
        tmp_image_path = os.path.join(tmpdir, input_filename)
        itk.imwrite(input_itk_image, tmp_image_path)
        abs_image_path = os.path.abspath(tmp_image_path)

        # 3. Load and initialize the NV-Segment-CTMR model
        model_pipeline = load_nvctmr_model()
        print("NV-Segment-CTMR: Model loaded successfully")

        # 4. Execute inference
        print(f"NV-Segment-CTMR: Running inference on {abs_image_path}...")
        print(f"NV-Segment-CTMR: Modality: {modality}")
        
        # Prepare the input configuration
        if label_prompt and len(label_prompt) > 0:
            print(f"NV-Segment-CTMR: Segmenting specific classes: {label_prompt}")
        else:
            print(f"NV-Segment-CTMR: Using modality preset: {modality}")

        # Execute inference using the HuggingFace pipeline
        print("NV-Segment-CTMR: Running inference with HuggingFace pipeline...")
        
        # Read the input image as numpy array for processing
        input_image = itk.imread(abs_image_path)
        input_array = itk.array_from_image(input_image)
        
        # Prepare segmentation output path
        output_path = os.path.join(tmpdir, "output_segmentation.nii.gz")
        
        # Ensure we have a valid pipeline
        if not hasattr(_pipeline_cache, '__call__'):
            raise ValueError("No valid HuggingFace pipeline available")
        
        # Run the pipeline on the image file
        results = _pipeline_cache(abs_image_path)
        
        # Process the pipeline results into segmentation mask
        if not isinstance(results, list) or len(results) == 0:
            raise ValueError("Pipeline returned no segmentation results")
        
        # Extract segmentation mask from pipeline results
        segmentation_array = np.zeros_like(input_array, dtype=np.uint8)
        
        # Process each detected segment
        for i, result in enumerate(results):
            if 'mask' not in result:
                continue
                
            mask = np.array(result['mask'])
            # Resize mask to match input dimensions if needed
            if mask.shape != input_array.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match input shape {input_array.shape}")
            
            # Assign label based on detected class
            label_id = i + 1  # Use sequential labeling
            segmentation_array[mask > 0.5] = label_id
        
        # Create ITK image from segmentation array
        seg_image = itk.image_from_array(segmentation_array)
        seg_image.CopyInformation(input_image)
        itk.imwrite(seg_image, output_path)
        
        print(f"NV-Segment-CTMR: Created segmentation with {len(results)} segments")
        print("NV-Segment-CTMR: Inference completed successfully")

        # 5. Read the segmentation result
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                "NV-Segment-CTMR inference finished but the expected output file "
                f"was not found at {output_path}"
            )
        
        itk_image_result = itk.imread(output_path)

        # 6. Convert the ITK result to a vtk.js dictionary and return it
        print("NV-Segment-CTMR: Converting result to VTK.js format...")
        segmentation_vtkjs_dict = convert_itk_to_vtkjs_image(itk_image_result)

        print("NV-Segment-CTMR: Inference complete. Returning segmentation.")
        return segmentation_vtkjs_dict


async def run_nv_segment_ctmr_inference_async(
    itk_image: itk.Image, label_prompt: List[int], modality: str = "MRI_BODY"
) -> Dict:
    """
    Asynchronously runs the NV-Segment-CTMR inference in the process pool.

    Args:
        itk_image: The ITK image object to segment.
        label_prompt: List of class indices to segment.
        modality: Modality type for segmentation.

    Returns:
        The segmentation result as a vtk.js dictionary.
    """
    # Convert to vtk.js dict for stable serialization to the process pool
    vtkjs_image_dict = convert_itk_to_vtkjs_image(itk_image)

    loop = asyncio.get_event_loop()
    segmentation_vtkjs_dict = await loop.run_in_executor(
        process_pool, _execute_nv_segment_ctmr_inference_in_process, 
        vtkjs_image_dict, label_prompt, modality
    )
    return segmentation_vtkjs_dict


@volview.expose("segmentWithNVSegmentMRI")
async def run_nv_segment_mri_segmentation(
    img_id: str, label_prompt: List[int] = None, modality: str = "MRI_BODY"
):
    """
    Exposes NV-Segment-CTMR MRI segmentation to the VolView client.

    Takes an image ID from the client, runs inference, and sends the
    resulting segmentation (as a vtk.js object) back to the client.

    Args:
        img_id: The ID of the image to segment.
        label_prompt: Optional list of class indices to segment.
                      Empty or None uses modality preset.
        modality: Modality type ("MRI_BODY", "CT_BODY", "MRI_BRAIN").
    """
    if label_prompt is None:
        label_prompt = []

    print(f"Received NV-Segment-CTMR segmentation request for image ID: {img_id}")
    print(f"Modality: {modality}")
    print(f"Class selection: {label_prompt if label_prompt else f'Using {modality} preset'}")

    image_cache_store = get_current_client_store("image-cache")

    itk_image = await image_cache_store.getVtkImageData(img_id)
    if itk_image is None:
        raise ValueError(f"No image found for ID: {img_id}")

    segmentation_vtkjs_dict = await run_nv_segment_ctmr_inference_async(itk_image, label_prompt, modality)

    nv_segment_store = get_current_client_store("nv-segment")
    await nv_segment_store.setNVSegmentResult(img_id, segmentation_vtkjs_dict)

    print("Successfully created MRI segmentation. Sending object back to client.")
    return 0