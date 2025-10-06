import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import itk
import numpy as np

def run_volview_insight_medgemma_inference(input_data: dict, itk_img: itk.image) -> str:
    """
    Runs inference using the MedGemma 27B - Multimodal model.

    Args:
        input_data (dict): A dictionary containing input data.
                           Expected to have a 'prompt' key with the user's question (str).
        itk_img (itk.image, optional): An ITK image object.

    Returns:
        str: The generated text response from the MedGemma model.
    """

    model_variant = "4b-it"  # @param ["4b-it", "27b-it", "27b-text-it"]
    model_id = f"google/medgemma-{model_variant}"
    is_thinking = False

    role_instruction = "You are an expert radiologist."
    # Max new tokens controls the length of the output response - how many tokens the LLM can generate. 
    if "27b" in model_variant and is_thinking:
        system_instruction = f"SYSTEM INSTRUCTION: think silently if needed. Please speak as an intelligent, concise physician short of time. {role_instruction}"
        max_new_tokens = 1300
    else:
        system_instruction = "Please speak as an intelligent, concise physician short of time." + role_instruction
        max_new_tokens = 300

    # Question
    user_question = input_data['prompt']

    # Load image 
    img_array = itk.array_from_image(itk_img).astype(int).squeeze()
    print('Input image array shape:', img_array.shape)
    image_uint8 = (255 * (img_array - img_array.min()) / (img_array.max() - img_array.min())).astype(np.uint8)
    image = Image.fromarray(image_uint8)

    prompt = f"Analyze the provided chest X-ray. Based on this data, answer the following question: {user_question}"  
    content = [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
    
    print('MedGemma prompt:', prompt)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": content
        }
    ]

    model_id = f"google/medgemma-{model_variant}"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # --- Start of per-request logic ---
    
    # Process inputs for the model
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    # Run inference in a memory-efficient context
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]

    # Decode the generated tokens into a string response
    response = processor.decode(generation, skip_special_tokens=True)

    # --- Memory Cleanup ---
    # This is the critical step to prevent CUDA memory errors on subsequent runs.
    # It explicitly deletes the large tensors from GPU memory.
    del inputs
    del generation
    torch.cuda.empty_cache()

    return response
