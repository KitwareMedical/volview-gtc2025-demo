import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import itk

from volview_insight_medgemma_inference import run_volview_insight_medgemma_inference
from volview_insight_clara_nv_reason_cxr_3b_inference import run_volview_insight_clara_nv_reason_cxr_3b_inference
from volview_server import VolViewApi, get_current_client_store
from volview_server.transformers import (
    convert_itk_to_vtkjs_image,
    convert_vtkjs_to_itk_image,
)

volview = VolViewApi()

process_pool = ProcessPoolExecutor(4)

def get_image_slice(img: itk.Image, active_layer: int | None = None) -> itk.Image:
    """If the image is 3D, extracts and returns the 2D slice  specified by active_layer.
    Otherwise, it assumes a 2D image and returns the input image.

    Args:
        img: The ITK image object.
        active_layer: The index of the 2D slice to process. If None, assumes 2D image.
    """
    # Check if the image is 3D and slicing is needed
    dimension = img.GetImageDimension()

    if active_layer is None:
        active_layer = 0
        
    if dimension == 2:
        slice_2d = img
    elif dimension == 3:
        # Set up extraction filter
        extract_filter = itk.ExtractImageFilter.New(img)
        extract_filter.SetDirectionCollapseToSubmatrix()

        # Define the extraction region
        input_region = img.GetBufferedRegion()
        size = input_region.GetSize()
        size[2] = 1  # Only one slice in Z
        start = input_region.GetIndex()
        start[2] = active_layer
        desired_region = input_region
        desired_region.SetSize(size)
        desired_region.SetIndex(start)

        extract_filter.SetExtractionRegion(desired_region)
        extract_filter.Update()
        slice_2d = extract_filter.GetOutput()
    else:
        raise RuntimeError("Input image has an invalid dimension")
    
    return slice_2d

def do_medgemma_inference(serialized_img: Dict[str, Any], analysis_input: Dict ) -> str:
    """Runs medGemma inference

    Args:
        serialized_img: The serialized ITK image (vtkjs format).
        analysis input: Dictionary containing the user query and parsed FHIR resource data

    Returns:
        The serialized text

    """
    itk_img = convert_vtkjs_to_itk_image(serialized_img)
    medgemma_response = run_volview_insight_medgemma_inference(input_data = analysis_input, itk_img = itk_img)

    return medgemma_response

def do_clara_nv_reason_cxr_3b_inference(serialized_img: Dict[str, Any], analysis_input: Dict) -> str:
    """Runs Clara NV-Reason-CXR-3B inference."""
    itk_img = convert_vtkjs_to_itk_image(serialized_img)
    response = run_volview_insight_clara_nv_reason_cxr_3b_inference(
        input_data=analysis_input, itk_img=itk_img
    )
    return response

INFERENCE_DISPATCH = {
    "MedGemma": do_medgemma_inference,
    "Clara NV-Reason-CXR-3B": do_clara_nv_reason_cxr_3b_inference,
}


@volview.expose("multimodalLlmAnalysis")
async def multimodal_llm_analysis(img_id: str | None = None, active_layer: int | None = None) -> None:
    """Runs multimodal LLM inference based on the selected model.

    Args:
        img_id: The ID of the image.
        active_layer: The index of the 2D slice to process. If None, assumes 2D image.
    """
    backend_store = get_current_client_store("backend-model-store")
    selected_model = await backend_store.selectedModel
    print(f"Starting multimodal LLM analysis with model: {selected_model}...")

    # --- 1. Get user prompt and vital signs data ---
    print("Got the backend model store. Fetching the analysis input dictionary...")
    analysis_input_dict = await backend_store.analysisInput[img_id]
    print(f"Got analysis input: {analysis_input_dict}")

    # --- 2. Get the appropriate inference function from the dispatch table ---
    inference_function = INFERENCE_DISPATCH.get(selected_model)
    if not inference_function:
        raise ValueError(f"Unknown model specified: '{selected_model}'. Available models: {list(INFERENCE_DISPATCH.keys())}")

    # --- 3. Get and process the image, if provided ---
    image_cache_store = get_current_client_store("image-cache")
    print("Got the images store. Fetching the image from the client...")
    img = await image_cache_store.getVtkImageData(img_id)
    print("Got the image data from the client. Starting image processing.")
    img_slice = get_image_slice(img, active_layer)
    serialized_img_vtkjs = convert_itk_to_vtkjs_image(img_slice)

    # --- 4. Execute the selected model's inference logic ---
    loop = asyncio.get_event_loop()
    try:
        model_response = await loop.run_in_executor(
            process_pool, inference_function, serialized_img_vtkjs, analysis_input_dict
        )
        await backend_store.setAnalysisResult(img_id, model_response)
        
        # Restore the final, detailed response log
        print(f"Analysis with {selected_model} finished. Response:\n{model_response}")

    except Exception as e:
        raise RuntimeError(
            f"Unexpected error during {selected_model} inference: {e}"
        ) from e
