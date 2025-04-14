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
print(f"Filtering for any of these labels: {candidate_labels}")

lines = []
for item in data:
    # Split normal_caption into individual labels
    item_labels = [label.strip() for label in item["normal_caption"].split(',')]
    
    # Check if any of the item's labels match any of our candidate labels
    if any(label in candidate_labels for label in item_labels):
        lines.append(item)

print(f"Found {len(lines)} entries matching at least one of the candidate labels")

with open("filtered_formatted_Class_QA.json", "w") as f:
    json.dump(lines, f, indent=4)
