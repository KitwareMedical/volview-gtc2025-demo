import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import itk
import numpy as np

def generate_vital_sign_summary_prompt(vital_signs_data: dict) -> str:
    """
    Parses a dictionary of vital sign measurements and generates a natural language
    summary prompt, focusing on the most recent measurement for each vital sign.

    Args:
        vital_signs_data (dict): A dictionary containing vital sign measurements.
                                 Expected keys include 'heart_rate', 'respiratory_rate',
                                 'spo2', 'systolic_bp', 'diastolic_bp' and
                                 'mean_arterial_pressure'. Each vital sign value is expected to be a list,
                                 with the most recent measurement as the last element.

    Returns:
        str: A natural language string summarizing the most recent vital signs
    """
    summary_parts = []

    # Define a mapping from dictionary keys to human-readable names and units
    vital_sign_map = {
        "heart_rate": {"name": "Heart Rate", "unit": "bpm"},
        "respiratory_rate": {"name": "Respiratory Rate", "unit": "breaths/min"},
        "spo2": {"name": "SpO2", "unit": "%"},
        "systolic_bp": {"name": "Systolic Blood Pressure", "unit": "mmHg"},
        "diastolic_bp": {"name": "Diastolic Blood Pressure", "unit": "mmHg"},
        "mean_arterial_pressure": {"name": "Mean Arterial Pressure", "unit": "mmHg"},
    }

    # Extract the most recent measurement for each vital sign
    for key, info in vital_sign_map.items():
        if key in vital_signs_data and vital_signs_data[key]:
            # The most recent measurement is the last element in the list
            most_recent_value = vital_signs_data[key][-1]
            summary_parts.append(f"{info['name']}: {most_recent_value} {info['unit']}")
        else:
            summary_parts.append(f"{info['name']}: Not available")

    # Combine the vital sign summaries
    vital_signs_summary = ", ".join(summary_parts)

    return vital_signs_summary

def run_volview_insight_medgemma_inference(input_data: dict, itk_img: itk.image = None) -> str:
    """
    Runs inference using the MedGemma 27B - Multimodal model. It can process either text-only prompts (not preferred)
    or prompts combined with an ITK image and vital signs data.

    Args:
        input_data (dict): A dictionary containing input data.
                           Expected to have a 'prompt' key with the user's question (str).
                           If an image is provided, we also check the input data for vital signs
                           data: ('heart_rate', 'respiratory_rate', 'spo2', 'systolic_bp',
                           'diastolic_bp', 'mean_arterial_pressure'), each with a list
                           of measurements ordered chronologically.
        itk_img (itk.image, optional): An ITK image object. If provided, the image
                                       will be processed along with the vital signs data
                                       and the prompt. Defaults to None.

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
    vital_signs_summary = generate_vital_sign_summary_prompt(input_data)

    if itk_img is not None:

        # Load image 
        img_array = itk.array_from_image(itk_img).astype(int).squeeze()
        print('Input image array shape:', img_array.shape)
        image_uint8 = (255 * (img_array - img_array.min()) / (img_array.max() - img_array.min())).astype(np.uint8)
        image = Image.fromarray(image_uint8)

        prompt = f"Analyze the provided chest X-ray and the patient's most recent vital signs: {vital_signs_summary}. Based on this data, answer the following question: {user_question}"  
        content = [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
    
    else:
        prompt = f"Analyze the patient's most recent vital signs: {vital_signs_summary}. Based on this data, answer the following question: {user_question}"  
        content = [
            {"type": "text", "text": prompt}
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
