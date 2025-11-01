import argparse
import os
import torch
import base64
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer
from PIL import Image

from word2png_function import text_to_images

CONFIG_EN_PATH = './config/config_en.json'
OUTPUT_DIR = './output_images'
INPUT_FILE = './input.txt'


def render_images():
    # Read text from file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    # import time
    # a = time.time()
    images = text_to_images(
        text=text,
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_EN_PATH,
        unique_id='render_test'
    )
    return images
    # b = time.time()
    # print(f"Time taken: {b - a} seconds")


def llm_vllm(_images):
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    prompt = "<image>\nFree OCR."
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    model_input = [{"prompt": f"{prompt}{text}"}]
    print(f"\nGenerated {len(_images)} image(s):")
    for img_path in _images:
        model_input.append({
            "prompt": prompt,
            "multi_modal_data": {"image": Image.open(img_path).convert("RGB")}
        })
        print(f"  {img_path}")

    # Create model instance
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor]
    )

    # Prepare batched input with your image file
    # image_1 = Image.open("path/to/your/image_1.png").convert("RGB")
    # image_2 = Image.open("path/to/your/image_2.png").convert("RGB")

    # model_input = [
    #     {
    #         "prompt": prompt,
    #         "multi_modal_data": {"image": image_1}
    #     },
    #     {
    #         "prompt": prompt,
    #         "multi_modal_data": {"image": image_2}
    #     }
    # ]

    sampling_param = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        # ngram logit processor args
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
        ),
        skip_special_tokens=False,
    )
    # Generate output
    model_outputs = llm.generate(model_input, sampling_param)

    # Print output
    total_tokens = 0
    with open("dsocr_output.txt", "w", encoding="utf-8") as f:
        for i, output in enumerate(model_outputs):
            # 关键点：从 output 对象中获取 prompt_token_ids 列表的长度
            num_input_tokens = len(output.prompt_token_ids)
            total_tokens += num_input_tokens

            print(f"\n[Request {i + 1}]")
            print(f"Input prompt: {output.prompt!r}")
            print(f"Number of Input Tokens: {num_input_tokens}")  # 打印输入 token 数量
            print(f"Generated Text: {output.outputs[0].text!r}")
            print("-" * 25)
            f.write(f"{output.outputs[0].text!r}")

    os.rename("dsocr_output.txt", f"dsocr_output_{total_tokens}.txt")


def llm_api(_images):
    client = OpenAI(base_url="YOUR_API_BASE_URL", api_key="YOUR_API_KEY")

    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    prompt_text = "Free OCR."  # The text part of your prompt

    # Process each image
    for i, img_path in enumerate(_images):
        base64_image = encode_image_to_base64(img_path)

        # Create the request payload according to your API's expected format
        # This structure mimics OpenAI's vision API format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"  # Or png, adjust accordingly
                        }
                    }
                ]
            }
        ]

        try:
            # Make the API call
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-OCR",  # Or the model name your API expects
                messages=messages,
                max_tokens=8192,
                temperature=0.0
                # Note: ngram logit processor equivalent might not be directly available via standard API
            )

            # Extract results
            generated_text = response.choices[0].message.content
            # Most OpenAI-compatible APIs return usage information
            prompt_tokens = response.usage.prompt_tokens if response.usage else "N/A"
            total_tokens = response.usage.total_tokens if response.usage else "N/A"

            print(f"\n[Request {i + 1}]")
            print(f"Input prompt text: {prompt_text!r}")
            print(f"Number of Prompt Tokens (from API): {prompt_tokens}")
            print(f"Total Tokens (from API): {total_tokens}")
            print(f"Generated Text: {generated_text!r}")
            print("-" * 25)

        except Exception as e:
            print(f"\n[Request {i + 1}] Error during API call: {e}")


def llm_hf(_images):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    model_name = 'deepseek-ai/DeepSeek-OCR'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True,
                                      use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)

    # prompt = "<image>\nFree OCR. "
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    # image_file = 'your_image.jpg'
    output_path = 'outputs/hf_results'

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

    # Tiny: base_size = 512, image_size = 512, crop_mode = False
    # Small: base_size = 640, image_size = 640, crop_mode = False
    # Base: base_size = 1024, image_size = 1024, crop_mode = False
    # Large: base_size = 1280, image_size = 1280, crop_mode = False

    # Gundam: base_size = 1024, image_size = 640, crop_mode = True
    for image_file in _images:
        model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path=output_path, base_size=1024,
                    image_size=640, crop_mode=True, save_results=True, test_compress=True)  # base


def main(mode='hf'):
    generated_images = render_images()
    # Collect generated image paths

    if mode == 'hf':
        print("\n--- LLM HF Inference ---")
        llm_hf(generated_images)
    elif mode == 'vllm':
        print("\n--- LLM VLLM Inference ---")
        llm_vllm(generated_images)
    elif mode == 'api':
        print("\n--- LLM API Inference ---")
        llm_api(generated_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upstream VLLM OCR Inference')
    parser.add_argument('--mode', type=str, default='hf', choices=['hf', 'vllm', 'api'],
                        help='Mode of LLM inference: hf (Hugging Face), vllm (VLLM), api (OpenAI-compatible API)')
    args = parser.parse_args()
    # mode = args.mode
    main(args.mode)
