import os
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Load Environment Variables
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "Missing API key. Set GROQ_API_KEY or OPENAI_API_KEY in .env."
    )

print("Loaded API key from environment.")

# -----------------------------
# Initialize Grow Client
# -----------------------------
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# -----------------------------
# Function
# -----------------------------
def generate_text_with_parameters(prompt, temp, top_p):
    """
    Generate text using different sampling parameters.
    """

    print("=" * 70)
    print(f"Temperature = {temp}")
    print(f"Top-P       = {top_p}")
    print("=" * 70)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful engineering assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temp,
            top_p=top_p,
            max_completion_tokens=150
        )

        output = response.choices[0].message.content

        print("\nGenerated Output:\n")
        print(output)
        print()

        return output

    except Exception as e:
        print(f"\nAPI Error:\n{e}")


# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":

    prompt = (
        "Write a three-sentence introduction for a novel about an "
        "autonomous robot studying ocean ecosystems."
    )

    print("\nRUN 1 : Deterministic Output\n")
    generate_text_with_parameters(
        prompt,
        temp=0.0,
        top_p=0.1
    )

    print("\nRUN 2 : Balanced Output\n")
    generate_text_with_parameters(
        prompt,
        temp=0.7,
        top_p=0.8
    )

    print("\nRUN 3 : Creative Output\n")
    generate_text_with_parameters(
        prompt,
        temp=1.2,
        top_p=0.95
    )