import json
import os
from groq import Groq

client = Groq()

input_path = os.path.join(
    "data", "d2_mcq", "processed", "cardiology_mcq_frozen.jsonl"
)
output_path = os.path.join(
    "data", "d2_mcq", "processed", "d2_mcq_verified_frozen.jsonl"
)

SYSTEM_PROMPT = """You are a medical classifier for a cardiology benchmark.
Analyze the multiple-choice question.

Task:
1. Is the PRIMARY topic or condition tested explicitly CARDIOVASCULAR medicine?
(Answer false if cardio terms like heart rate, pulse, or hypertension are merely incidental background vitals, or if the main diagnosis is renal, neurological, psychiatric, or infectious).
2. If true, assign it to exactly ONE category:
   - Structural Heart Disease
   - Functional Heart Disease
   - Coronary & Vascular Disease
   - Special Heart Diseases

Return strictly valid JSON:
{"is_cardio": true, "category": "Coronary & Vascular Disease"}
or
{"is_cardio": false, "category": null}
"""

with open(input_path, "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

verified_items = []
print(f"Classifying {len(items)} questions via Groq Llama 3.1...")

for idx, item in enumerate(items):
    user_prompt = f"Question: {item['question']}\nOptions: {item['options']}\nAnswer: {item['answer']}"
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        res_json = json.loads(response.choices[0].message.content)
        if res_json.get("is_cardio"):
            item["category"] = res_json.get("category")
            verified_items.append(item)

        if (idx + 1) % 25 == 0:
            print(
                f"Processed {idx + 1}/{len(items)} | Cardio kept: {len(verified_items)}"
            )

    except Exception as e:
        print(f"Skipping {item['id']}: {e}")

with open(output_path, "w", encoding="utf-8") as f:
    for item in verified_items:
        f.write(json.dumps(item) + "\n")

print(f"\nFinished! Kept {len(verified_items)} clean items in {output_path}")