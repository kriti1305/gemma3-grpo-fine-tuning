# 🧠 LLM Reward Lab

### Instruction-Tuned LLM Response Generation & Reward Evaluation

[🚀 **Live Demo**](https://gemma3-grpo-fine-tuning-wh4jyjtqxrtqxpkhircqdy.streamlit.app/)

An interactive LLM application that generates responses using an instruction-tuned language model and evaluates them using a multi-signal reward scoring system.

## 🚀 Features

- Interactive LLM response generation
- Instruction-tuned language model
- GRPO-style reward evaluation
- Multi-signal response scoring
- Relevance evaluation
- Response-length evaluation
- Formatting evaluation
- Streamlit web interface
- CPU inference support

## 🧠 How It Works

1. User enters a prompt.
2. The language model generates a response.
3. The response is evaluated using reward functions.
4. Relevance, length, and formatting signals are calculated.
5. A combined reward score is displayed.

## ⭐ Reward Evaluation

The reward system evaluates responses using:

- **Length Reward** – checks whether the response has a reasonable length.
- **Relevance Reward** – checks whether the response contains relevant keywords.
- **Format Reward** – evaluates basic readability and formatting.

The final reward is calculated by combining these signals.

## 🛠️ Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- PEFT / LoRA
- Pytest

## 🧪 Testing

The project includes automated tests for the reward functions.

**Test Result:** 5 tests passed successfully.

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
