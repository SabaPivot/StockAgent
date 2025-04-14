import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import logging

logger = logging.getLogger(__name__)

class XrayVQADataset(Dataset):
    """Dataset for X-ray images, questions, and their corresponding answers from JSON"""
    def __init__(self, image_root, json_file, processor, tokenizer, img_size, max_q_len=128, max_a_len=512):
        """
        Args:
            image_root (str): Path to the directory containing images.
            json_file (str): Path to the JSON file containing the data triplets.
            processor: Vision processor for images.
            tokenizer: Language model tokenizer.
            img_size (int): Target size to resize images to (e.g., 384).
            max_q_len (int): Maximum token length for the question ('problem').
            max_a_len (int): Maximum token length for the answer ('normal_caption').
        """
        self.image_root = image_root
        self.img_size = img_size
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len

        # Ensure tokenizer has a pad token; set to EOS if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set tokenizer pad_token to eos_token ({self.tokenizer.eos_token})")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.samples = json.load(f)
            logger.info(f"Loaded {len(self.samples)} samples from {json_file}")
        except FileNotFoundError:
            logger.error(f"JSON file not found at {json_file}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {json_file}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred loading JSON: {e}")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing processed image, tokenized question,
        and tokenized answer suitable for Stage 2 training.
        """
        try:
            sample = self.samples[idx]
            image_filename = sample.get("image")
            question_text = sample.get("problem")
            answer_text = sample.get("normal_caption") # Using 'normal_caption' as the answer

            if not all([image_filename, question_text, answer_text]):
                logger.warning(f"Sample {idx} is missing required fields (image, problem, or normal_caption). Skipping.")
                return self.__getitem__((idx + 1) % len(self)) # Recursively get next valid item

            image_path = os.path.join(self.image_root, image_filename)

            # --- Image Processing ---
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.img_size, self.img_size))
            # Note: processor likely adds batch dim, remove it here, add back in dataloader collate_fn if needed
            image_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = image_inputs.pixel_values.squeeze(0)

            # --- Text Tokenization ---
            # Tokenize Question (problem) - no padding needed here usually, as it's concatenated
            question_tokens = self.tokenizer(
                question_text,
                max_length=self.max_q_len,
                truncation=True,
                add_special_tokens=False # Often don't add BOS/EOS here, handled during concatenation
            ).input_ids

            # Tokenize Answer (normal_caption) - requires padding and label masking
            answer_tokens_full = self.tokenizer(
                answer_text,
                max_length=self.max_a_len,
                padding="max_length", # Pad answers to max length
                truncation=True,
                return_tensors="pt" # Get tensors directly
            )
            answer_input_ids = answer_tokens_full.input_ids.squeeze(0) # Remove batch dim

            # Create labels: shift tokens right, replace padding with -100
            # In standard Causal LM training, labels are input_ids shifted right.
            # We will handle the full sequence concatenation and label creation in the Trainer.
            # Here, we just provide the raw token IDs for question and answer.

            # The trainer will concatenate: [VISUAL_TOKENS] + [QUESTION_TOKENS] + [ANSWER_TOKENS]
            # The labels will be: [-100] * len(VISUAL+QUESTION) + [ANSWER_TOKENS] (with padding as -100)

            return {
                "pixel_values": pixel_values,
                "question_input_ids": torch.tensor(question_tokens, dtype=torch.long), # Tensor format
                "answer_input_ids": answer_input_ids, # Already a tensor, padded
            }

        except FileNotFoundError:
            logger.warning(f"Image file not found for sample {idx}: {image_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            logger.error(f"Error processing sample {idx} ({image_path}): {e}", exc_info=True)
            return self.__getitem__((idx + 1) % len(self))

# Example Usage (Optional: For testing the dataset)
if __name__ == '__main__':
    from transformers import AutoProcessor, AutoTokenizer
    import tempfile
    import shutil

    # --- Create dummy data for testing ---
    print("Setting up dummy data for testing...")
    dummy_root = tempfile.mkdtemp()
    dummy_json_path = os.path.join(dummy_root, "dummy_data.json")
    dummy_image_dir = os.path.join(dummy_root, "images")
    os.makedirs(dummy_image_dir, exist_ok=True)

    # Create a dummy image
    dummy_img_path = os.path.join(dummy_image_dir, "test_img.png")
    try:
        dummy_image = Image.new('RGB', (60, 30), color = 'red')
        dummy_image.save(dummy_img_path)
    except Exception as e:
        print(f"Warning: Could not create dummy image using PIL: {e}")


    dummy_data = [
        {"image": "test_img.png", "problem": "What is shown in the image?", "normal_caption": "A red rectangle."},
        {"image": "test_img.png", "problem": "Describe the color.", "normal_caption": "The color is red."},
        {"image": "non_existent.png", "problem": "This should be skipped", "normal_caption": "Skipped"}, # Test missing image
         {"image": "test_img.png", "problem": "Long question " * 20, "normal_caption": "Long answer " * 50}, # Test truncation
    ]
    with open(dummy_json_path, 'w') as f:
        json.dump(dummy_data, f)

    # --- Initialize dummy processor and tokenizer ---
    # Using small models for quick testing
    # Using CLIP processor as a placeholder for SigLIP-like processor
    print("Initializing processor and tokenizer...")
    try:
        # Use a standard vision processor (like CLIP or SigLIP's)
        # Replace with your actual vision model name if different
        vision_model_name = "openai/clip-vit-base-patch32"
        # Use a standard causal LM tokenizer (like Llama or Gemma's)
        # Replace with your actual LLM name
        llm_name = "gpt2" #"google/gemma-2b" # gemma might require login

        processor = AutoProcessor.from_pretrained(vision_model_name)
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        # Set pad token if missing (GPT2 example)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set GPT2 pad token to EOS: {tokenizer.pad_token}")


        # --- Test Dataset Initialization ---
        print("Initializing Dataset...")
        dataset = XrayVQADataset(
            image_root=dummy_image_dir,
            json_file=dummy_json_path,
            processor=processor,
            tokenizer=tokenizer,
            img_size=224, # Use processor's default size or specify
            max_q_len=32,
            max_a_len=64
        )

        # --- Test __len__ ---
        print(f"Dataset length: {len(dataset)}")
        assert len(dataset) == len(dummy_data), f"Expected length {len(dummy_data)}, got {len(dataset)}"

        # --- Test __getitem__ ---
        print("Fetching first valid item...")
        item = dataset[0] # Get the first valid item
        print("Item keys:", item.keys())
        assert "pixel_values" in item
        assert "question_input_ids" in item
        assert "answer_input_ids" in item
        print("Pixel values shape:", item["pixel_values"].shape)
        print("Question tokens:", item["question_input_ids"])
        print("Answer tokens:", item["answer_input_ids"])
        print("Decoded Question:", tokenizer.decode(item["question_input_ids"]))
        # Decode answer, skipping padding tokens for clarity
        decoded_answer = tokenizer.decode(item["answer_input_ids"][item["answer_input_ids"] != tokenizer.pad_token_id])
        print("Decoded Answer (no padding):", decoded_answer)

        # --- Test Dataloader ---
        print("Testing with DataLoader...")
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        print("Batch keys:", batch.keys())
        print("Batch pixel_values shape:", batch["pixel_values"].shape)
        print("Batch question_input_ids shape:", batch["question_input_ids"].shape)
        print("Batch answer_input_ids shape:", batch["answer_input_ids"].shape)

        print("Dataset test completed successfully.")

    except ImportError as e:
         print(f"Import error during testing: {e}. Make sure transformers and Pillow are installed.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        # Clean up dummy data
        print("Cleaning up dummy data...")
        shutil.rmtree(dummy_root)
        print("Dummy data cleaned up.") 