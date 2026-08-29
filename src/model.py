from transformers import pipeline

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model():
    """Load an instruction-tuned language model for CPU inference."""

    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        device=-1,
    )


def generate_response(generator, prompt):
    """Generate a useful instruction-following response."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and knowledgeable AI assistant. "
                "Answer the user's question clearly and accurately. "
                "If you are unsure about something, say so instead of "
                "making up information."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    result = generator(
        messages,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        num_return_sequences=1,
    )

    generated = result[0]["generated_text"]

    # Extract the assistant's answer from the chat response.
    if isinstance(generated, list):
        for message in reversed(generated):
            if message.get("role") == "assistant":
                return message.get("content", "").strip()

    return str(generated).strip()


def main():
    print("=" * 60)
    print("LOCAL INSTRUCTION-TUNED LLM")
    print("=" * 60)

    print("\nLoading model...")
    generator = load_model()

    print("Model loaded successfully!")
    print("\nType 'exit' to stop.\n")

    while True:
        prompt = input("You: ").strip()

        if prompt.lower() == "exit":
            print("\nExiting...")
            break

        if not prompt:
            print("Please enter a prompt.")
            continue

        response = generate_response(
            generator,
            prompt
        )

        print(f"\nModel: {response}\n")


if __name__ == "__main__":
    main()