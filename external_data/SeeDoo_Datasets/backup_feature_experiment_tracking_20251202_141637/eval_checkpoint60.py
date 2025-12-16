import json, csv, torch
from pathlib import Path
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

print("Loading models...")
qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", quantization_config=qconfig, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
ft_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", quantization_config=qconfig, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
ft_model = PeftModel.from_pretrained(ft_model, "saves/qwen2vl-treadmill-lora-pipeline/checkpoint-60")
ft_model.eval()

with open("/app/data/preformat_right_only_test_20251130_142614.json", "r") as f:
    test_data = json.load(f)

csv_path = "/app/evaluation_results_20251130_124835/detailed_predictions_checkpoint60.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["model", "video", "full_prediction", "label"])
    writer.writeheader()
    print(f"Evaluating {len(test_data)} videos...")
    for i, item in enumerate(test_data):
        vpath = "/app/" + item["videos"][0]
        vname = Path(vpath).name
        label = "moving" if "moving" in next(m for m in item["messages"] if m["role"]=="assistant")["content"].lower() else "stopped"
        msgs = [{"role":"user","content":[{"type":"video","video":vpath,"fps":4.0,"min_pixels":224*224,"max_pixels":384*384},{"type":"text","text":"Analyze this video. Is the treadmill belt moving or stopped?"}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(msgs)
        inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")

        with torch.no_grad():
            # Base model - get FULL OUTPUT
            inp_b = {k:v.to(base_model.device) for k,v in inputs.items()}
            gen_ids = base_model.generate(**inp_b, max_new_tokens=128, do_sample=False)
            trim = [o[len(i):] for i,o in zip(inp_b["input_ids"], gen_ids)]
            base_full_output = processor.batch_decode(trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # Fine-tuned model - get FULL OUTPUT
            inp_f = {k:v.to(ft_model.device) for k,v in inputs.items()}
            gen_ids = ft_model.generate(**inp_f, max_new_tokens=128, do_sample=False)
            trim = [o[len(i):] for i,o in zip(inp_f["input_ids"], gen_ids)]
            ft_full_output = processor.batch_decode(trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        writer.writerow({"model":"base","video":vname,"full_prediction":base_full_output,"label":label})
        writer.writerow({"model":"finetuned","video":vname,"full_prediction":ft_full_output,"label":label})
        if (i+1)%10==0: print(f"Processed {i+1}/{len(test_data)}")

print(f"\nCSV saved: {csv_path}")
print(f"\nFirst 10 rows:")
with open(csv_path, encoding="utf-8") as f:
    for i,line in enumerate(f):
        if i<10: print(line.rstrip())
