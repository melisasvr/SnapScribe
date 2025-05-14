import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import time
import os
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("📸 AI Image Captioning")
st.markdown("""
This application uses a pre-trained deep learning model to generate captions for your images.
Simply upload an image and get an AI-generated description!
""")

# Initialize the model and processor
@st.cache_resource
def load_model():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Social media caption enhancements
def enhance_for_social(caption, style="instagram"):
    """Make captions more engaging for social media"""
    
    # Dictionary of popular hashtags by category
    hashtags = {
        "beach": ["#beachlife", "#oceanvibes", "#vitaminsea", "#paradise", "#beachday"],
        "food": ["#foodie", "#yummy", "#delicious", "#foodporn", "#instafood"],
        "nature": ["#naturelovers", "#outdoors", "#beautifulnature", "#wilderness", "#earthfocus"],
        "city": ["#citylife", "#urbanphotography", "#streetphotography", "#citylights", "#skyline"],
        "people": ["#portraitphotography", "#smile", "#happy", "#moments", "#lifestyle"],
        "pets": ["#petsofinstagram", "#dogsofinstagram", "#catsofinstagram", "#cutepets", "#furbaby"],
        "sunset": ["#sunset", "#goldenhour", "#sunsetlovers", "#magichour", "#dusk"],
        "travel": ["#wanderlust", "#travelgram", "#exploremore", "#adventure", "#travelphotography"]
    }
    
    # Emoji mapping
    emojis = {
        "beach": ["🏖️", "🌊", "🏝️", "☀️", "🐚"],
        "food": ["😋", "🍽️", "👩‍🍳", "🍕", "🍰"],
        "nature": ["🌿", "🌳", "🌸", "🏞️", "🍃"],
        "city": ["🌆", "🏙️", "🌃", "🚕", "🌉"],
        "people": ["😊", "👋", "🤗", "👫", "💫"],
        "pets": ["🐶", "🐱", "🐾", "💕", "🥰"],
        "sunset": ["🌅", "🌄", "✨", "💛", "🧡"],
        "travel": ["✈️", "🧳", "🗺️", "🌍", "🚗"]
    }
    
    # Determine content type based on keywords in caption
    content_type = "nature"  # default
    for category in hashtags.keys():
        if category in caption.lower():
            content_type = category
            break
    
    # Get random hashtags and emojis for the content type
    import random
    selected_hashtags = random.sample(hashtags[content_type], min(3, len(hashtags[content_type])))
    selected_emojis = random.sample(emojis[content_type], min(2, len(emojis[content_type])))
    
    # Enhance caption based on style
    if style == "instagram":
        # Make first letter capitalized and add period if needed
        caption = caption[0].upper() + caption[1:]
        if not caption.endswith((".", "!", "?")):
            caption += "."
            
        # Add emojis and creative flair
        enhanced = f"{selected_emojis[0]} {caption} {selected_emojis[-1]}\n\n"
        
        # Add engaging question or statement
        engagement_phrases = [
            "Who else loves this view?",
            "Double tap if this makes you smile!",
            "Can you imagine being here right now?",
            "Tag someone who needs to see this!",
            "This moment is everything!",
            "Living for these moments!"
        ]
        enhanced += random.choice(engagement_phrases) + "\n\n"
        
        # Add hashtags
        enhanced += " ".join(selected_hashtags)
        
    elif style == "twitter":
        # Shorter, punchier caption
        enhanced = f"{caption} {' '.join(selected_emojis)}\n\n"
        enhanced += random.choice(selected_hashtags)
        
    else:  # default
        enhanced = caption
        
    return enhanced

# Function to generate captions
def generate_caption(image, processor, model, use_advanced=False, social_style=None, temperature=1.0):
    try:
        # Convert PIL Image to RGB (in case it's RGBA)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Process the image
        inputs = processor(image, return_tensors="pt")
        
        # Generate the caption
        if use_advanced:
            # Use beam search for better quality captions
            output = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True,
                num_return_sequences=3,
                temperature=temperature
            )
            # Return multiple captions
            captions = []
            for out in output:
                caption = processor.decode(out, skip_special_tokens=True)
                # Apply social media enhancement if requested
                if social_style:
                    caption = enhance_for_social(caption, social_style)
                captions.append(caption)
            return captions
        else:
            # Standard generation
            output = model.generate(**inputs, max_length=30, temperature=temperature)
            caption = processor.decode(output[0], skip_special_tokens=True)
            # Apply social media enhancement if requested
            if social_style:
                caption = enhance_for_social(caption, social_style)
            return [caption]
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return ["Error generating caption. Please try again."]

# Function to create placeholder images
def create_placeholder_image(width, height, color=(100, 150, 200)):
    # Create a solid color image as a placeholder
    return Image.fromarray(np.full((height, width, 3), color, dtype=np.uint8))

# Function to load sample images
def load_sample_images():
    samples = []
    
    # Sample 1: Nature placeholder
    nature_img = create_placeholder_image(400, 300, (100, 150, 200))
    samples.append((nature_img, "A landscape photo"))
    
    # Sample 2: Portrait placeholder
    portrait_img = create_placeholder_image(400, 300, (150, 100, 150))
    samples.append((portrait_img, "A portrait photo"))
    
    # Sample 3: Food placeholder
    food_img = create_placeholder_image(400, 300, (200, 150, 100))
    samples.append((food_img, "A food photo"))
    
    return samples

# Main function
def main():
    # Load the model
    with st.spinner("Loading model..."):
        processor, model = load_model()
        
    if processor is None or model is None:
        st.error("Failed to load the image captioning model. Please refresh the page or try again later.")
        return
    
    # Sidebar options
    st.sidebar.header("Options")
    use_advanced = st.sidebar.checkbox("Generate multiple detailed captions", value=False)
    
    # Social media styling options
    st.sidebar.subheader("Social Media Style")
    social_style = st.sidebar.radio(
        "Choose a caption style:",
        [None, "instagram", "twitter"],
        format_func=lambda x: "Standard" if x is None else x.capitalize(),
        index=0
    )
    
    if social_style:
        st.sidebar.info(f"Using {social_style.capitalize()} style with hashtags and emojis!")
    
    # Caption creativity settings
    st.sidebar.subheader("Creativity Settings")
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.5, 
        max_value=1.5, 
        value=1.0, 
        step=0.1,
        help="Higher values make captions more creative but less accurate"
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Generate caption
            with st.spinner("Generating captions..."):
                start_time = time.time()
                captions = generate_caption(image, processor, model, use_advanced, social_style, temperature)
                end_time = time.time()
            
            # Display results
            with col2:
                st.subheader("Generated Captions:")
                for i, caption in enumerate(captions):
                    caption_container = st.container()
                    with caption_container:
                        st.markdown(f"**Caption {i+1}:**")
                        st.markdown(f"{caption}")
                        if social_style:
                            # Add copy button
                            if st.button(f"📋 Copy Caption {i+1}", key=f"copy_{i}"):
                                # This doesn't actually copy to clipboard in Streamlit,
                                # but we can show the text in a way that's easy to copy
                                st.code(caption, language=None)
                                st.success("Caption ready to copy! ✅")
                
                st.text(f"Processing time: {end_time - start_time:.2f} seconds")
        except Exception as e:
            st.error(f"Error processing image: {e}")

    # Additional information
    st.markdown("---")
    with st.expander("How It Works", expanded=False):
        st.markdown("""
        This application uses the BLIP (Bootstrapping Language-Image Pre-training) model developed by Salesforce Research. 
        The model combines vision and language understanding to generate natural language descriptions of images.
        
        ### Social Media Caption Enhancement
        
        When you select a social media style (Instagram or Twitter), the app:
        1. Analyzes the base caption to identify content themes
        2. Adds relevant emojis and hashtags
        3. Includes engagement prompts for Instagram
        4. Formats the caption appropriately for each platform
        
        ### Applications
        
        - **Social Media**: Generate engaging captions for Instagram, Twitter, and other platforms
        - **Accessibility**: Help visually impaired users understand image content
        - **Content Creation**: Automate caption generation for blogs and websites
        - **Image Search**: Enable searching images by their content description
        - **Data Organization**: Automatically tag and categorize image collections
        """)

    # Display examples of social media captions
    if social_style:
        st.markdown("---")
        st.subheader("Creative Caption Examples")
        
        example_captions = {
            "instagram": [
                "🏖️ Golden sands stretch as far as the eye can see, with crystal clear waters inviting you in. 🌊\n\nTag someone who needs a beach day right now!\n\n#beachlife #vitaminsea #paradise",
                "🍕 Freshly baked pizza with bubbling cheese and the perfect crispy crust. 🍽️\n\nDouble tap if this makes you hungry!\n\n#foodie #yummy #delicious",
                "🌿 Sunlight filtering through ancient trees, creating a magical forest atmosphere. ✨\n\nWho else loves getting lost in nature?\n\n#naturelovers #outdoors #earthfocus"
            ],
            "twitter": [
                "Life's better with sand between your toes and salt in the air 🏖️ 🌊\n\n#beachlife",
                "When the sunset paints the sky in shades you didn't even know existed ✨ 🌅\n\n#goldenhour",
                "City lights and urban heights - finding beauty in the concrete jungle 🌃 🏙️\n\n#citylife"
            ]
        }
        
        if social_style in example_captions:
            for example in example_captions[social_style]:
                st.markdown(f"_{example}_")
                st.markdown("---")

    # Display potential issues section
    st.markdown("---")
    st.subheader("Troubleshooting")
    st.markdown("""
    - If the model is slow to load, it's downloading required files. This only happens on first run.
    - For better results, use clear, well-lit images.
    - The model works best with common objects and scenes it was trained on.
    """)

if __name__ == "__main__":
    main()