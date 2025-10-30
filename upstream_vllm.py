from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

from word2png_function import text_to_images

CONFIG_EN_PATH = './config/config_en.json'
OUTPUT_DIR = './output_images'
INPUT_FILE = './input.txt'

# Read text from file
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    text = f.read()
# import time
# a = time.time()
images = text_to_images(
    text= text,
    output_dir=OUTPUT_DIR,
    config_path=CONFIG_EN_PATH,
    unique_id='Little_Red_Riding_Hood'
)
# b = time.time()
# print(f"Time taken: {b - a} seconds")

prompt = "<image>\nFree OCR."
model_input = []
print(f"\nGenerated {len(images)} image(s):")
for img_path in images:
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
for i, output in enumerate(model_outputs):
    # 关键点：从 output 对象中获取 prompt_token_ids 列表的长度
    num_input_tokens = len(output.prompt_token_ids)

    print(f"\n[Request {i + 1}]")
    print(f"Input prompt: {output.prompt!r}")
    print(f"Number of Input Tokens: {num_input_tokens}")  # 打印输入 token 数量
    print(f"Generated Text: {output.outputs[0].text!r}")
    print("-" * 25)