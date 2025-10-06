import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import os
from typing import Any, Dict

import itk
import numpy as np
from sklearn.linear_model import LinearRegression

from volview_insight_seg_inference import run_volview_insight_seg_inference
from volview_insight_medgemma_inference import run_volview_insight_medgemma_inference
from volview_insight_clara_nv_reason_cxr_3b_inference import run_volview_insight_clara_nv_reason_cxr_3b_inference
from volview_server import VolViewApi, get_current_client_store, get_current_session
from volview_server.transformers import (
    convert_itk_to_vtkjs_image,
    convert_vtkjs_to_itk_image,
)

## Link to app ##


volview = VolViewApi()


## median filter example ##
# copied from
# https://github.com/Kitware/VolView/blob/411e5a891bfb520647ab3f97cac6edfcca930a65/server/examples/example_api.py

process_pool = ProcessPoolExecutor(4)


@dataclass
class ClientState:
    image_id_map: dict = field(init=False, default_factory=dict)
    blurred_ids: set = field(init=False, default_factory=set)


def associate_images(state: ClientState, image_id: str, blurred_id: str) -> None:
    """Associates original and blurred image IDs in the client state.

    Args:
        state: The current client state object.
        image_id: The ID of the original image.
        blurred_id: The ID of the blurred image.
    """
    state.blurred_ids.add(blurred_id)
    state.image_id_map[image_id] = blurred_id
    state.image_id_map[blurred_id] = image_id


def get_base_image(state: ClientState, img_id: str) -> str:
    """Gets the original image ID from a potentially blurred image ID.

    Args:
        state: The current client state object.
        img_id: The ID of the image (can be original or blurred).

    Returns:
        The ID of the original image.
    """
    if img_id in state.blurred_ids:
        return state.image_id_map[img_id]
    return img_id


async def show_image(img_id: str) -> None:
    """Sets the primary selection in VolView to display an image.

    Args:
        img_id: The ID of the image to display.
    """
    store = get_current_client_store("dataset")
    await store.setPrimarySelection({"type": "image", "dataID": img_id})

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

@volview.expose("exampleAnalysis")
async def example_analysis(patient_id: str) -> None:
    """Performs a linear regression analysis on patient data.

    Args:
        patient_id: The ID of the patient.
    """
    print(f"Started example analysis...")
    # constructs a descriptor to point to the client for future retrieval of
    # resources. All operations on it are cached untill awaited.. 
    store = get_current_client_store("python-analysis-store")

    # If not using await below, it would not do anything. All attribute accesses
    # would construct a chain of accessors and descriptors for the data,
    # triggered by await.
    analysis_input = await store.analysisInput[patient_id]
    print(f"Got the input... ({analysis_input})")

    points = np.array(analysis_input)
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression()
    model.fit(x, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(x, y)

    await store.setAnalysisResult(patient_id, [slope, intercept, r_squared])

    print(f"Example analysis finished and got result [{slope}, {intercept}, {r_squared}]")

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
async def multimodal_llm_analysis(patient_id: str, img_id: str | None = None, active_layer: int | None = None) -> None:
    """Runs multimodal LLM inference based on the selected model.

    If an img_id is specified, the corresponding image and patient's vital signs
    records are used by the model to respond to the user's prompt.

    Args:
        patient_id: The ID of the patient.
        img_id: The ID of the image to provide to the model.
        active_layer: The index of the 2D slice to process. If None, assumes 2D image.
    """
    backend_store = get_current_client_store("backend-model-store")
    selected_model = await backend_store.selectedModel
    print(f"Starting multimodal LLM analysis with model: {selected_model}...")

    # --- 1. Get user prompt and vital signs data ---
    print("Got the backend model store. Fetching the analysis input dictionary...")
    analysis_input_dict = await backend_store.analysisInput[patient_id]
    print(f"Got analysis input: {analysis_input_dict}")

    # --- 2. Get the appropriate inference function from the dispatch table ---
    inference_function = INFERENCE_DISPATCH.get(selected_model)
    if not inference_function:
        raise ValueError(f"Unknown model specified: '{selected_model}'. Available models: {list(INFERENCE_DISPATCH.keys())}")

    # --- 3. Get and process the image, if provided ---
    serialized_img_vtkjs = None
    if img_id is not None:
        image_store = get_current_client_store("images")
        print("Got the images store. Fetching the image from the client...")
        state = get_current_session(default_factory=ClientState)
        base_image_id = get_base_image(state, img_id)
        img = await image_store.dataIndex[base_image_id]
        print("Got the image data from the client. Starting image processing.")
        img_slice = get_image_slice(img, active_layer)
        serialized_img_vtkjs = convert_itk_to_vtkjs_image(img_slice)
    else:
        print(f"Analysis with {selected_model} did not get an image ID. Will proceed without image.")

    # --- 4. Execute the selected model's inference logic ---
    loop = asyncio.get_event_loop()
    try:
        model_response = await loop.run_in_executor(
            process_pool, inference_function, serialized_img_vtkjs, analysis_input_dict
        )
        await backend_store.setAnalysisResult(patient_id, model_response)
        
        # Restore the final, detailed response log
        print(f"Analysis with {selected_model} finished. Response:\n{model_response}")

    except Exception as e:
        raise RuntimeError(
            f"Unexpected error during {selected_model} inference: {e}"
        ) from e

def do_lung_segmentation(serialized_img: Dict[str, Any]) -> Dict[str, Any]:
    """Performs lung segmentation on a serialized image using a pre-trained model.

    Args:
        serialized_img: The serialized ITK image (vtkjs format).

    Returns:
        The serialized output segmentation image.

    Raises:
        FileNotFoundError: If the segmentation model file is not found.
    """
    model_path = './segmentLungsModel-v1.0.ckpt'

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please ensure that the model has been downloaded and placed in the correct location. "
            "Refer to the README for instructions on obtaining and installing the model."
        )

    itk_img = convert_vtkjs_to_itk_image(serialized_img)
    seg = run_volview_insight_seg_inference(itk_img, model_path)
    return convert_itk_to_vtkjs_image(seg)


async def run_lung_segmentation_process(img: itk.Image, active_layer: int | None = None) -> itk.Image:
    """Runs the lung segmentation in a separate process to avoid blocking the GIL.

    Args:
        img: The ITK image object.
        active_layer: The index of the 2D slice to process. If None, assumes 2D image.

    Returns:
        The ITK image object after segmentation.

    Raises:
        RuntimeError: If the segmentation model is missing or an unexpected error occurs.
    """
    print(f"Layer/slice index `{active_layer}` chosen for segmentation...")

    slice_2d = get_image_slice(img, active_layer)

    # ==== Processing logic ====
    serialized_img_vtkjs = convert_itk_to_vtkjs_image(slice_2d)
    loop = asyncio.get_event_loop()

    try:
        serialized_output = await loop.run_in_executor(
            process_pool, do_lung_segmentation, serialized_img_vtkjs
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Lung segmentation failed due to missing model file: {e}"
        ) from None  # suppress chained traceback
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error during lung segmentation: {e}"
        ) from e

    processed_slice = convert_vtkjs_to_itk_image(serialized_output)
    # ==========================

    # Determine where to paste
    input_region = img.GetBufferedRegion()
    start = list(input_region.GetIndex())
    start[2] = active_layer  # Put slice back at the right Z

    # If image was originally 3D, paste the processed slice back
    if img.GetImageDimension() == 3 and active_layer is not None:
        paste_filter = itk.PasteImageFilter.New(img)
        paste_filter.SetSourceImage(processed_slice)
        paste_filter.SetDestinationImage(img)
        paste_filter.SetDestinationIndex(start)
        paste_filter.SetSourceRegion(processed_slice.GetBufferedRegion())
        paste_filter.Update()
        return paste_filter.GetOutput()
    else:
        return processed_slice


async def getDataID(imageID: str) -> str:
    """Retrieves the volume key for a given image ID from the DICOM store.

    Args:
        imageID: The ID of the image.

    Returns:
        The volume key if found, otherwise the original image ID.
    """
    dicomStore = get_current_client_store('dicom')
    volumeKey = dicomStore.imageIDToVolumeKey[imageID]
    return volumeKey if volumeKey else imageID

@volview.expose("segmentLungs")
async def segment_lungs(img_id: str, active_layer: int | None = None) -> None:
    """Performs lung segmentation on an image in VolView.

    If the input image is already processed, it re-runs the operation
    on the original image.

    Args:
        img_id: The ID of the image to segment.
        active_layer: If a multi-layer (3D) image is passed, which layer to be
            used for the (2D) segmentation. If None, it's a 2D image.
    """
    print(f"Started segmentLungs on VolView \"images\" store image ID: {img_id} ...")
    store = get_current_client_store("images")
    # layerStore = get_current_client_store("layer")
    state = get_current_session(default_factory=ClientState)

    # Behavior: when a filter request occurs on a
    # processed image, we instead assume we are re-running
    # the operation on the original image.
    base_image_id = get_base_image(state, img_id)
    img = await store.dataIndex[base_image_id]

    # we need to run the filter in a subprocess,
    # since itk blocks the GIL.
    segout = await run_lung_segmentation_process(img, active_layer)
    print(f"Completed segmentLungs on VolView \"images\" store image ID: {img_id}.")

    seg_id = state.image_id_map.get(base_image_id)
    seg_exists_on_client_side = None
    if seg_id:
        seg_exists_on_client_side = await store.metadata[seg_id]
        # seg_exists_on_client_side = await store.dataIndex[seg_id]

    print('seg_exists_on_client_side: ', seg_exists_on_client_side)

    if seg_id and seg_exists_on_client_side:
        print(f"Updating existing segmentation image ID: {seg_id}.")
        await store.updateData(seg_id, segout)
    else:
        seg_id = await store.addVTKImageData(f"{img_id}_seg", segout)
        print(f"New segmentation image ID: {seg_id}.")
        # Associate the segmented image ID with the base image ID.
        associate_images(state, base_image_id, seg_id)
        # volumeKey = getDataID()
        # parent = { 'type': 'dicom', 'dataID': await getDataID(f'{img_id}') }
        # source = { 'type': 'image', 'dataID': seg_id }
        # await layerStore.addLayer(parent, source)
