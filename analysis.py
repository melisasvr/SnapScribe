import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Advanced Image Captioning Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Advanced Image Captioning Analysis")
st.markdown("""
This dashboard allows you to analyze the performance and compare different caption generation approaches
for the BLIP image captioning model.
""")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_captions_with_params(image, processor, model, params):
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    inputs = processor(image, return_tensors="pt")
    
    # Generate with specific parameters
    output = model.generate(
        **inputs,
        max_length=params['max_length'],
        num_beams=params['num_beams'],
        early_stopping=params['early_stopping'],
        num_return_sequences=params['num_return_sequences'],
        temperature=params['temperature'],
        do_sample=params['do_sample']
    )
    
    captions = []
    for out in output:
        caption = processor.decode(out, skip_special_tokens=True)
        captions.append(caption)
    
    return captions

def create_placeholder_image(width, height, color=(100, 150, 200)):
    return Image.fromarray(np.full((height, width, 3), color, dtype=np.uint8))

def main():
    processor, model = load_model()
    
    # Sidebar - Parameter selection
    st.sidebar.header("Model Parameters")
    
    max_length = st.sidebar.slider("Max Length", min_value=10, max_value=100, value=30, 
                                  help="Maximum length of generated caption")
    
    num_beams = st.sidebar.slider("Beam Size", min_value=1, max_value=10, value=5,
                                 help="Number of beams for beam search")
    
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                                   help="Controls randomness. Lower = more deterministic")
    
    do_sample = st.sidebar.checkbox("Use Sampling", value=False, 
                                   help="Whether to use sampling; use greedy decoding otherwise")
    
    early_stopping = st.sidebar.checkbox("Early Stopping", value=True,
                                        help="Whether to stop beam search when enough sequences finish")
    
    num_return_sequences = st.sidebar.slider("Number of Captions", min_value=1, max_value=5, value=3,
                                           help="Number of captions to generate")
    
    # Create parameter sets for comparison
    params_default = {
        'max_length': 30,
        'num_beams': 5,
        'early_stopping': True,
        'num_return_sequences': 3,
        'temperature': 1.0,
        'do_sample': False
    }
    
    params_custom = {
        'max_length': max_length,
        'num_beams': num_beams,
        'early_stopping': early_stopping,
        'num_return_sequences': num_return_sequences,
        'temperature': temperature,
        'do_sample': do_sample
    }
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            tabs = st.tabs(["Default Parameters", "Custom Parameters", "Comparison"])
            
            with tabs[0]:
                with st.spinner("Generating captions with default parameters..."):
                    start_time = time.time()
                    default_captions = generate_captions_with_params(image, processor, model, params_default)
                    default_time = time.time() - start_time
                
                st.subheader("Captions with Default Parameters")
                for i, caption in enumerate(default_captions):
                    st.markdown(f"**Caption {i+1}:** {caption}")
                st.text(f"Processing time: {default_time:.2f} seconds")
            
            with tabs[1]:
                with st.spinner("Generating captions with custom parameters..."):
                    start_time = time.time()
                    custom_captions = generate_captions_with_params(image, processor, model, params_custom)
                    custom_time = time.time() - start_time
                
                st.subheader("Captions with Custom Parameters")
                for i, caption in enumerate(custom_captions):
                    st.markdown(f"**Caption {i+1}:** {caption}")
                st.text(f"Processing time: {custom_time:.2f} seconds")
            
            with tabs[2]:
                st.subheader("Parameter Comparison")
                
                # Create comparison table
                data = {
                    "Parameter": ["Max Length", "Beam Size", "Temperature", "Do Sample", "Early Stopping", "# Sequences"],
                    "Default": [params_default['max_length'], params_default['num_beams'], 
                               params_default['temperature'], params_default['do_sample'],
                               params_default['early_stopping'], params_default['num_return_sequences']],
                    "Custom": [params_custom['max_length'], params_custom['num_beams'], 
                              params_custom['temperature'], params_custom['do_sample'],
                              params_custom['early_stopping'], params_custom['num_return_sequences']]
                }
                
                st.table(data)
                
                # Time comparison chart
                times = [default_time, custom_time]
                labels = ["Default", "Custom"]
                
                fig, ax = plt.subplots()
                ax.bar(labels, times)
                ax.set_ylabel('Processing Time (seconds)')
                ax.set_title('Processing Time Comparison')
                st.pyplot(fig)
    
    else:
        # Show example image options
        st.markdown("### Try with an example image:")
        
        example_cols = st.columns(3)
        
        # Create sample images
        sample1 = create_placeholder_image(400, 300, (100, 150, 200))
        sample2 = create_placeholder_image(400, 300, (150, 100, 150))
        sample3 = create_placeholder_image(400, 300, (200, 150, 100))
        
        samples = [
            (sample1, "Landscape Image"),
            (sample2, "Portrait Image"),
            (sample3, "Food Image")
        ]
        
        for i, ((sample, desc), col) in enumerate(zip(samples, example_cols)):
            with col:
                st.image(sample, caption=desc, use_column_width=True)
                if st.button(f"Analyze Example {i+1}"):
                    st.session_state.selected_sample = sample
                    st.experimental_rerun()
        
        if 'selected_sample' in st.session_state:
            image = st.session_state.selected_sample
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption="Selected Example", use_column_width=True)
            
            with col2:
                with st.spinner("Generating captions..."):
                    start_time = time.time()
                    default_captions = generate_captions_with_params(image, processor, model, params_default)
                    default_time = time.time() - start_time
                    
                    start_time = time.time()
                    custom_captions = generate_captions_with_params(image, processor, model, params_custom)
                    custom_time = time.time() - start_time
                
                st.subheader("Default Parameters")
                for i, caption in enumerate(default_captions):
                    st.markdown(f"**Caption {i+1}:** {caption}")
                    
                st.subheader("Custom Parameters")
                for i, caption in enumerate(custom_captions):
                    st.markdown(f"**Caption {i+1}:** {caption}")
    
    # Explanation section
    st.markdown("---")
    st.header("Understanding the Parameters")
    
    st.markdown("""
    ### Max Length
    Controls the maximum number of tokens (roughly words) in the generated caption.
    
    ### Beam Size
    The number of paths to explore during beam search. Higher values often lead to better quality but slower generation.
    
    ### Temperature
    Controls randomness in the generation process. Lower values make the output more deterministic.
    
    ### Do Sample
    When enabled, uses sampling instead of greedy decoding, introducing more variability.
    
    ### Early Stopping
    Stops the beam search when enough full sequences have been generated.
    
    ### Number of Captions
    How many different caption variations to generate.
    """)
    
if __name__ == "__main__":
    main()