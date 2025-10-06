import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import itk
import numpy as np
from typing import Dict, Any

def run_volview_insight_clara_nv_reason_cxr_3b_inference(input_data: Dict[str, Any], itk_img: itk.Image) -> str:
    """
    Runs inference using the local nvidia-reason-cxr-3b model on a chest X-ray.
    This version automatically detects and uses available hardware (GPU or CPU).

    Args:
        input_data (Dict[str, Any]): A dictionary containing the user's prompt
                                     under the 'prompt' key.
        itk_img (itk.Image): An ITK image object of the chest X-ray.

    Returns:
        str: The generated text response from the model.
    """
    if itk_img is None:
        raise ValueError("Nvidia CXR model requires an image for analysis.")

    # --- 1. Define Model Path and Load ---
    model_path = "models/nvidia-reason-cxr-3b"

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    # --- 2. Prepare Image ---
    img_array = itk.array_view_from_image(itk_img).squeeze()
    # Normalize to 8-bit integer if not already
    if img_array.dtype != np.uint8:
        numerator = img_array - img_array.min()
        denominator = img_array.max() - img_array.min()
        # Handle the case of a flat image (denominator is zero)
        if denominator > 0:
            img_array = (255 * numerator / denominator).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)
    image = Image.fromarray(img_array).convert("RGB")

    # --- 3. Prepare Input Prompt ---
    user_question = input_data.get('prompt', "Find abnormalities and support devices.")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_question}]}]

    # --- 4. Process Inputs for Model ---
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    # The .to(model.device) ensures tensors are on the same device as the model
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

    # --- 5. Generate Response ---
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    # --- 6. Trim and Decode Output ---
    input_ids_len = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, input_ids_len:]
    generated_text = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # --- 7. Clean Up Memory ---
    del inputs, generated_ids, model, processor
    # Only try to empty cache if CUDA is available, preventing errors on CPU-only machines.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Analysis with {model_path} finished. Response:\n{generated_text}")
    return generated_text
