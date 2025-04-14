import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_labels", type=str, required=True, help="Comma-separated list of candidate labels (e.g., 'Atelectasis, No Finding')")
    return parser.parse_args()

with open("/mnt/WHY/VLM/Deepseek/VLM-R1/QA_DATASET/Test/SQA/formatted_Class_QA.json", "r") as f:
    data = json.load(f)

args = parse_args()
# Split candidate labels into a list, stripping whitespace
candidate_labels = [label.strip() for label in args.candidate_labels.split(',')]
print(f"Filtering for EXACT matches to: {candidate_labels}")

lines = []
for item in data:
    # Only include if normal_caption is EXACTLY one of the candidate labels
    if item["normal_caption"] in candidate_labels:
        lines.append(item)

print(f"Found {len(lines)} entries with exact matches to candidate labels")

with open("filtered_formatted_Class_QA.json", "w") as f:
    json.dump(lines, f, indent=4)
