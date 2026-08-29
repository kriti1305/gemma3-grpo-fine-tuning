import streamlit as st

from src.model import load_model, generate_response
from src.rewards import combined_reward


st.set_page_config(
    page_title="LLM Reward Optimization",
    page_icon="🧠",
    layout="centered",
)


@st.cache_resource
def load_llm():
    return load_model()


st.title("🧠 LLM Reward Optimization System")

st.markdown(
    """
    ### GRPO-Style Response Evaluation

    Generate a response using a local language model and
    evaluate its quality using a reward-based scoring system.
    """
)

st.divider()

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Example: Explain machine learning",
    height=120,
)

if st.button("🚀 Generate Response", use_container_width=True):

    if not prompt.strip():
        st.warning("Please enter a prompt first.")

    else:
        with st.spinner("Generating response..."):

            generator = load_llm()

            response = generate_response(
                generator,
                prompt,
            )

            keywords = [
                word.strip(".,?!")
                for word in prompt.split()
                if len(word.strip(".,?!")) > 3
            ]

            reward = combined_reward(
                response=response,
                keywords=keywords,
            )

        st.subheader("🤖 Model Response")

        st.write(response)

        st.subheader("⭐ Reward Score")

        st.metric(
            label="Response Quality",
            value=f"{reward:.4f}",
        )

        st.progress(float(reward))

        st.caption(
            "The reward score combines relevance, formatting, "
            "and response-length signals."
        )