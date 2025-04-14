import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_labels", type=str, required=True, help="Comma-separated list of candidate labels (e.g., 'Atelectasis, No Finding')")
    parser.add_argument("--output_path", type=str, default="/mnt/samuel/Siglip/filtered_formatted_Class_QA.json", 
                      help="Path to save the filtered JSON file")
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

# Ensure the output directory exists
output_dir = os.path.dirname(args.output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save to the specified output path
with open(args.output_path, 'w') as f:
    print(f"Saving filtered data to: {args.output_path}")
    json.dump(lines, f, indent=4)
