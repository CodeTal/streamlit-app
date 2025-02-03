import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import os.path 
import pickle

hf_token = "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"

@st.cache_resource
def load_model():
    base_model_path = 'meta-llama/Llama-3.2-3B-Instruct'
    lora_model_path = os.path.join("checkpoints", "llm_checkpoints", "dpo_video_and_content_instruct_beta=0.1_r=32_new_guideline_ckpt4000")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, use_auth_token=hf_token)
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_auth_token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'

    return tokenizer, model

tokenizer, model = load_model()

guideline_path = "./checkpoints/guideline_extraction_outputs/llama_3.2_3b_epoch2_iteration50.pkl"
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
video_summary = st.text_area("Enter the summary of the article's accompanying video (optional):", placeholder="Type the summary of the video related to the article...")

if st.button("Generate Headline"):
    if content.strip():
        if not video_summary.strip():
            video_summary = ''
        prompt = process_prompt(tokenizer, content, video_summary, guidelines)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        st.write("### Generated 5 Potential Headlines:")
        
        # Generate 5 headlines in parallel
        outputs = model.generate(**inputs,
                               max_new_tokens=60,
                               num_return_sequences=5,
                               do_sample=True, 
                               temperature=0.7)
        
        # Process all outputs
        for i, output in enumerate(outputs):
            st.write(f"### Headline {i+1}")
            response = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = response.replace('"', '')
            st.write(f"{response}")
    else:
        st.write("Please enter a valid prompt.")
