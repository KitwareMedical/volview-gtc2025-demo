import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import itk
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Global model cache to avoid reloading on every inference
_model_cache: Optional[Tuple[AutoModelForImageTextToText, AutoProcessor]] = None
_model_path = "/Users/andrew.howe/projects/vvi/2025-nvidiagtcdc-vvi/server/2025_nvidiagtcdc/models/nvidia-reason-cxr-3b"

def get_model_and_processor() -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load and cache the model and processor. Reuses cached instances if available."""
    global _model_cache

    if _model_cache is None:
        print(f"Loading Clara model from {_model_path} (first time)...")
        model = AutoModelForImageTextToText.from_pretrained(
            _model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(_model_path, use_fast=True, local_files_only=True)
        _model_cache = (model, processor)
        print("Clara model loaded and cached successfully.")

    return _model_cache

def run_volview_insight_clara_nv_reason_cxr_3b_inference(input_data: Dict[str, Any], itk_img: itk.Image) -> str:
    """
    Runs inference using the local nvidia-reason-cxr-3b model on a chest X-ray.
    This version automatically detects and uses available hardware (GPU or CPU).

    Args:
        input_data (Dict[str, Any]): A dictionary containing the user's prompt
                                     under the 'prompt' key and conversation
                                     history under the 'history' key.
        itk_img (itk.Image): An ITK image object of the chest X-ray.

    Returns:
        str: The generated text response from the model.
    """
    if itk_img is None:
        raise ValueError("Nvidia CXR model requires an image for analysis.")

    # --- 1. Get Model and Processor (cached) ---
    model, processor = get_model_and_processor()

    # --- 2. Prepare Image ---
    img_array = itk.array_view_from_image(itk_img).squeeze()
    print(f"Original image shape: {img_array.shape}, dtype: {img_array.dtype}")

    # Normalize to 8-bit integer if not already
    if img_array.dtype != np.uint8:
        numerator = img_array - img_array.min()
        denominator = img_array.max() - img_array.min()
        # Handle the case of a flat image (denominator is zero)
        if denominator > 0:
            img_array = (255 * numerator / denominator).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)

    print(f"After normalization - shape: {img_array.shape}, dtype: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")
    image = Image.fromarray(img_array).convert("RGB")
    print(f"PIL Image created - size: {image.size}, mode: {image.mode}")

    # --- 3. Prepare Input Prompt and History ---
    user_question = input_data.get('prompt', "Find abnormalities and support devices.")
    history = input_data.get('history', [])

    # Build messages from history
    messages = []
    if len(history) > 0:
        valid_index = None
        for i in range(len(history)):
            h = history[i]
            # Check if the content is non-empty
            if len(h.get('content', '').strip()) > 0:
                # Find the first assistant message to determine where to start
                if valid_index is None and h['role'] == 'assistant':
                    valid_index = i - 1
                messages.append({
                    "role": h['role'],
                    "content": [{"type": "text", "text": h['content']}]
                })

        # Remove previous messages without image if needed
        if valid_index is None:
            messages = []
        if len(messages) > 0 and valid_index > 0:
            messages = messages[valid_index:]  # Keep only messages from the first assistant response onwards

    # Add current user prompt
    messages.append({"role": "user", "content": [{"type": "text", "text": user_question}]})

    # Always insert the image at the beginning of the FIRST user message
    # This is critical for the model to have image context throughout the conversation
    messages[0]['content'].insert(0, {"type": "image"})

    print(f"Messages for Clara CXR: {messages}")

    # --- 4. Process Inputs for Model ---
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    # Pass image directly without 'image' key in content
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(model.device)

    # --- 5. Generate Response ---
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)

    # --- 6. Trim and Decode Output ---
    input_ids_len = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, input_ids_len:]
    generated_text = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # --- 7. Clean Up Memory (but keep model cached) ---
    del inputs, generated_ids
    # Only try to empty cache if CUDA is available, preventing errors on CPU-only machines.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Analysis with {_model_path} finished. Response:\n{generated_text}")
    return generated_text
