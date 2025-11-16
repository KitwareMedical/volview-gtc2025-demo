import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import itk
import numpy as np

from nv_reason_cxr_inference import run_nv_reason_cxr_inference_streaming, get_model_and_processor
from volview_server import VolViewApi, get_current_client_store
from volview_server.transformers import (
    convert_itk_to_vtkjs_image,
    convert_vtkjs_to_itk_image,
)

volview = VolViewApi()

# Thread pool for LLM inference (1 worker to avoid multiple model instances)
# PyTorch releases the GIL during inference, so threads work well here
llm_thread_pool = ThreadPoolExecutor(max_workers=1)


# --- Server Startup: Preload Model ---
def preload_models():
    """
    Preload the NV-Reason model during server startup.
    This prevents the first client request from timing out during model loading.
    """
    print("\n" + "="*70)
    print("PRELOADING MODELS ON SERVER STARTUP")
    print("="*70)
    print("This may take ~1 minute on first run...")

    import time
    start_time = time.time()

    try:
        # Load the model - this will cache it for all future requests
        get_model_and_processor()
        elapsed = time.time() - start_time
        print(f"\n✅ Model preloaded successfully in {elapsed:.1f}s")
        print(f"Server is ready to accept requests!")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\n❌ Failed to preload model: {e}")
        print("Server will attempt to load on first request instead.")
        print("="*70 + "\n")


# Preload models when this module is imported (server startup)
preload_models()


def convert_vtkjs_rgb_to_grayscale(vtkjs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a multi-component vtkjs image to grayscale by extracting the first component.

    This is needed for NV-Reason-CXR which expects grayscale input. The generic ITK
    transformer doesn't handle multi-component images, so we preprocess here.

    Args:
        vtkjs_dict: A vtkjs image dictionary

    Returns:
        A modified vtkjs dictionary with single-component grayscale data
    """
    pixel_data_array = vtkjs_dict["pointData"]["arrays"][0]["data"]
    num_components = pixel_data_array.get("numberOfComponents", 1)

    # If already grayscale, return as-is
    if num_components == 1:
        return vtkjs_dict

    # Extract first component for multi-component images (RGB → grayscale)
    pixel_values = pixel_data_array["values"]

    # Convert to numpy array, extract first component, convert back
    if isinstance(pixel_values, bytes):
        # Determine dtype from dataType field
        dtype_map = {
            "Uint8Array": np.uint8,
            "Int8Array": np.int8,
            "Uint16Array": np.uint16,
            "Int16Array": np.int16,
            "Uint32Array": np.uint32,
            "Int32Array": np.int32,
            "Float32Array": np.float32,
            "Float64Array": np.float64,
        }
        dtype = dtype_map.get(pixel_data_array["dataType"], np.uint8)
        pixel_array = np.frombuffer(pixel_values, dtype=dtype)
    else:
        pixel_array = np.array(pixel_values)

    # Extract every nth value (first component only)
    # e.g., for RGB: [R0,G0,B0,R1,G1,B1,...] -> [R0,R1,...]
    grayscale_array = pixel_array[::num_components]

    # Create a new vtkjs dict with single-component data
    result = vtkjs_dict.copy()
    result["pointData"] = {
        **vtkjs_dict["pointData"],
        "arrays": [
            {
                **vtkjs_dict["pointData"]["arrays"][0],
                "data": {
                    **pixel_data_array,
                    "numberOfComponents": 1,
                    "size": len(grayscale_array),
                    "values": grayscale_array.tobytes(),
                }
            }
        ]
    }

    return result

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

def do_clara_nv_reason_cxr_3b_inference_streaming(itk_img: itk.Image, analysis_input: Dict):
    """Runs Clara NV-Reason-CXR-3B inference with streaming.

    Note: Model warmup happens in the parent endpoint before this is called
    to avoid stream timeouts during first load.
    
    Args:
        itk_img: ITK image (already converted from vtkjs by RPC deserializer)
        analysis_input: Dictionary with 'prompt' and 'history' keys
    """
    return run_nv_reason_cxr_inference_streaming(
        input_data=analysis_input, itk_img=itk_img
    )


STREAMING_INFERENCE_DISPATCH = {
    "Clara NV-Reason-CXR-3B": do_clara_nv_reason_cxr_3b_inference_streaming,
}



@volview.expose("multimodalLlmAnalysisStream")
async def multimodal_llm_analysis_stream(img_id: str | None = None, active_layer: int | None = None):
    """Runs multimodal LLM inference with streaming support.

    Args:
        img_id: The ID of the image.
        active_layer: The index of the 2D slice to process. If None, assumes 2D image.

    Yields:
        dict: Streaming token data in the format {"token": str}
    """
    import time
    start_time = time.time()

    backend_store = get_current_client_store("backend-model-store")
    selected_model = await backend_store.selectedModel
    print(f"Starting streaming analysis with {selected_model} (image: {img_id}, layer: {active_layer})")

    # --- 1. Get user prompt ---
    analysis_input_dict = await backend_store.analysisInput[img_id]

    # --- 2. Get the appropriate inference function from the dispatch table ---
    inference_function = STREAMING_INFERENCE_DISPATCH.get(selected_model)
    if not inference_function:
        raise ValueError(f"Unknown model specified: '{selected_model}'. Available models: {list(STREAMING_INFERENCE_DISPATCH.keys())}")

    # --- 3. Get and process the image ---
    # Note: We disable auto-deserialization (transform_args=False) to get the raw
    # vtkjs format from the client, then manually convert to ITK. This makes the
    # API behavior match its name (getVtkImageData returns vtkjs data).
    image_cache_store = get_current_client_store("image-cache", transform_args=False)
    vtkjs_data = await image_cache_store.getVtkImageData(img_id)
    itk_img = convert_vtkjs_to_itk_image(vtkjs_data)
    
    img_slice = get_image_slice(itk_img, active_layer)

    # --- 4. Execute streaming inference ---
    # Note: Model is preloaded during server startup, so this should be fast
    full_response = ""
    token_count = 0
    streaming_start = time.time()

    try:
        # Pass the ITK image slice and analysis input to the inference function
        token_generator = inference_function(img_slice, analysis_input_dict)

        # Yield tokens as they arrive
        async for token in token_generator:
            full_response += token
            token_count += 1

            # Log first token
            if token_count == 1:
                first_token_time = time.time() - streaming_start
                print(f"First token after {first_token_time:.1f}s, streaming started...")

            yield {"token": token}

        streaming_time = time.time() - streaming_start
        total_time = time.time() - start_time

        # Store the complete response in the backend store
        await backend_store.setAnalysisResult(img_id, full_response)

        # Verify XML structure
        has_think = '<think>' in full_response
        has_answer = '<answer>' in full_response

        print(f"✅ Stream complete: {token_count} tokens in {streaming_time:.1f}s "
              f"({token_count / streaming_time:.1f} tok/s) | "
              f"<think>: {has_think}, <answer>: {has_answer}")

    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Stream error after {error_time:.1f}s ({token_count} tokens): {type(e).__name__}: {e}")

        # Log partial response for debugging
        if full_response:
            print(f"Partial response: {full_response[:100]}...")

        # Re-raise with context
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"Error during {selected_model} streaming inference: {e}"
        ) from e
