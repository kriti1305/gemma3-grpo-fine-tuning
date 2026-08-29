import streamlit as st

from src.model import load_model, generate_response
from src.rewards import (
    combined_reward,
    length_reward,
    relevance_reward,
    format_reward,
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="LLM Reward Lab",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .model-card {
        padding: 18px;
        border-radius: 12px;
        background-color: #f5f7fa;
        border: 1px solid #e1e5ea;
        margin-bottom: 20px;
    }

    .response-card {
        padding: 22px;
        border-radius: 12px;
        background-color: #f8f9fb;
        border: 1px solid #e1e5ea;
        margin-top: 10px;
    }

    .score-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e1e5ea;
        text-align: center;
        margin-top: 10px;
    }

    .score {
        font-size: 42px;
        font-weight: 700;
    }

    .small-text {
        color: #666;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

@st.cache_resource
def load_llm():
    return load_model()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 LLM Reward Lab</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Instruction-Tuned LLM Response Generation & Reward Evaluation'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

st.markdown(
    """
    <div class="model-card">
        <b>🤖 Language Model</b><br>
        Qwen2.5-1.5B-Instruct
        <br><br>
        <b>⚙️ Execution</b><br>
        Local CPU Inference
        <br><br>
        <b>⭐ Evaluation</b><br>
        Multi-signal reward scoring
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Prompt section
# ---------------------------------------------------------

st.subheader("💬 Ask the Model")

prompt = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: Explain machine learning in simple terms."
    ),
    height=130,
    label_visibility="collapsed",
)


# Example prompts

st.caption("Try an example:")

example_columns = st.columns(3)

with example_columns[0]:
    if st.button(
        "🧠 What is AI?",
        use_container_width=True,
    ):
        prompt = "What is artificial intelligence?"

with example_columns[1]:
    if st.button(
        "📊 Explain machine learning",
        use_container_width=True,
    ):
        prompt = "Explain machine learning in simple terms."

with example_columns[2]:
    if st.button(
        "💻 SQL vs NoSQL",
        use_container_width=True,
    ):
        prompt = "What is the difference between SQL and NoSQL?"


st.write("")


generate = st.button(
    "🚀 Generate & Evaluate Response",
    use_container_width=True,
    type="primary",
)


# ---------------------------------------------------------
# Generate response
# ---------------------------------------------------------

if generate:

    if not prompt.strip():

        st.warning(
            "Please enter a question before generating a response."
        )

    else:

        with st.spinner(
            "Loading the model and generating your response..."
        ):

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

            # Individual reward components
            length_score = length_reward(response)
            relevance_score = relevance_reward(
                response,
                keywords,
            )
            format_score = format_reward(response)

            # Final reward
            reward = combined_reward(
                response=response,
                keywords=keywords,
            )


        st.divider()

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        st.subheader("🤖 Model Response")

        st.markdown(
            f"""
            <div class="response-card">
                {response}
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.write("")


        # -------------------------------------------------
        # Reward score
        # -------------------------------------------------

        st.subheader("⭐ Response Quality")

        score_columns = st.columns([1, 2])

        with score_columns[0]:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="small-text">
                        Overall Reward
                    </div>
                    <div class="score">
                        {reward:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with score_columns[1]:

            st.write("")

            st.progress(
                min(max(float(reward), 0.0), 1.0)
            )

            st.caption(
                "Overall reward score ranges from 0 to 1."
            )


        # -------------------------------------------------
        # Reward breakdown
        # -------------------------------------------------

        st.subheader("📊 Reward Breakdown")

        reward_columns = st.columns(3)

        with reward_columns[0]:

            st.metric(
                "📏 Length",
                f"{length_score:.2f}",
            )

        with reward_columns[1]:

            st.metric(
                "🎯 Relevance",
                f"{relevance_score:.2f}",
            )

        with reward_columns[2]:

            st.metric(
                "📝 Formatting",
                f"{format_score:.2f}",
            )


        st.caption(
            "Final reward = 30% Length + 50% Relevance + "
            "20% Formatting"
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "LLM Reward Lab • Local inference • "
    "GRPO-style reward evaluation"
)