# describe_scene.py
import subprocess
from datetime import datetime

PROMPT_TEMPLATE = """You are describing a short moment from a Unity game clip.

You are given a CSV where each row is an object and each column Frame_1 … Frame_N is that object's (x, y, z) position over time. Higher y ≈ higher in the air; changes in x/y/z mean movement. Some objects may not move and can be ignored.

Your task:
1) First paragraph: briefly describe what the scene feels like overall — what seems still, what might be active, and how the space feels (low/high, near/far). Keep it plain and direct.
2) Second paragraph: describe how things change across the frames — who/what starts still, what begins to move, which direction movement happens in, and how it ends.

Rules:
- Use simple, factual, and calm language.
- Do NOT use emotional or dramatic words.
- Do NOT mention numbers, coordinates, frames, axes, or technical terms.
- Write exactly two short paragraphs, less than 120 words total.
- The tone should feel like an audio description: neutral, clear, and steady.

Here is the CSV:
"""

def describe_with_ollama(cleaned_csv_path="object_positions_slime_cleaned.csv"):
    # Read cleaned CSV
    with open(cleaned_csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()

    # Combine the exact prompt with the CSV content
    full_prompt = PROMPT_TEMPLATE + "\n" + csv_content

    print("🧩 Sending prompt to Ollama (qwen2.5-coder:7b)...\n")

    # Run Ollama with UTF-8-safe pipes
    result = subprocess.run(
        ["ollama", "run", "qwen2.5-coder:7b"],
        input=full_prompt.encode("utf-8"),  # encode before sending
        capture_output=True
    )

    # Decode output safely
    response = result.stdout.decode("utf-8", errors="replace").strip()
    print("🗒️ Model Response:\n")
    print(response)

    # Save response to a timestamped text file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"scene_description_{timestamp}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)

    print(f"\n✅ Saved model output to: {output_file}")

if __name__ == "__main__":
    describe_with_ollama()
