from src.model import load_model, generate_response
from src.rewards import combined_reward


def main():
    print("=" * 60)
    print("LLM + GRPO REWARD EVALUATION")
    print("=" * 60)

    print("\nLoading language model...")
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

        response = generate_response(generator, prompt)

        # Use important words from the prompt as simple
        # relevance keywords.
        keywords = [
            word.strip(".,?!")
            for word in prompt.split()
            if len(word.strip(".,?!")) > 3
        ]

        reward = combined_reward(
            response=response,
            keywords=keywords,
        )

        print(f"\nModel: {response}")
        print(f"Reward Score: {reward:.4f}\n")


if __name__ == "__main__":
    main()