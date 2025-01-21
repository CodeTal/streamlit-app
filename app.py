import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pickle


# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "checkpoints/llm_checkpoints/dpo_video_and_content_instruct_beta=0.1_r=32_guideline"
    model = AutoModelForCausalLM.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'

    return tokenizer, model

tokenizer, model = load_model()

guideline_path = "checkpoints/guideline_extraction_outputs/whole_training_data/llama_3.2_3b_epoch2.pkl"
guideline_results = pickle.load(open(guideline_path,'rb'))
guidelines = guideline_results['best_rational']

def process_prompt(tokenizer, content, video_summary = '', guidelines = None):
    if guidelines:
        system_prompt = "You are a helpful assistant that writes engaging headlines. To maximize engagement, you may follow these proven guidelines:\n" + guidelines
    else:
        system_prompt = "You are a helpful assistant that writes engaging headlines."

    user_prompt = (
        f"Below is an article and its accompanying video summary:\n\n"
        f"Article Content:\n{content}\n\n"
        f"Video Summary:\n{'None' if video_summary == '' else video_summary}\n\n"
        f"Write ONLY a single engaging headline that accurately reflects the article. Do not include any additional text, explanations, or options."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


st.title("Article Headline Writer")
st.write("Write a catchy headline from content and video summary.")

# Inputs for content and video summary
content = st.text_area("Enter the article content:", placeholder="Type the main content of the article here...")
video_summary = st.text_area("Enter the video summary:", placeholder="Type the summary of the video related to the article...")

# guideline = st.checkbox("Use Guidelines", value=True)
if st.button("Generate Headline"):
    if content.strip():
        if not video_summary.strip():
            video_summary = ''
        prompt = process_prompt(tokenizer, content, video_summary, guidelines)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        outputs = model.generate(**inputs, 
                               max_length=60, 
                               num_return_sequences=5,
                               do_sample=True,
                               temperature=0.7)
        
        st.write("### Generated Headlines:")
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            st.write(f"{i+1}. {response}")
    else:
        st.write("Please enter a valid prompt.")
