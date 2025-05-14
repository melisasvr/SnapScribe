# SnapScribe: AI-Powered Image Captioning
SnapScribe is an intelligent image captioning application that generates descriptive and creative captions for your images using state-of-the-art AI technology. Perfect for social media creators, content managers, and accessibility enhancement.
## ✨ Features
- Automatic Caption Generation: Transform any image into descriptive text with AI
- Social Media Enhancement: Generate platform-specific captions for Instagram and Twitter
- Multiple Caption Options: Get various creative descriptions for the same image
- Customizable Creativity: Adjust settings to control caption style and creativity
- Easy-to-Use Interface: Simple, intuitive Streamlit-based UI

## 🛠️ Tech Stack
- Framework: Streamlit
- AI Model: BLIP (Bootstrapping Language-Image Pre-training)
- Libraries: PyTorch, Transformers, PIL, NumPy

## 📋 Requirements
- Python 3.7+
- PyTorch
- Transformers
- Streamlit
- PIL
- NumPy

## 🚀 Installation & Setup
1. Clone the repository:
- `bashgit clone https://github.com/yourusername/snapscribe.git`
- `cd snapscribe`
2. Create and activate a virtual environment (optional but recommended):
- `bashpython -m venv venv`
- `source venv/bin/activate  # On Windows: venv\Scripts\activate`

## Install dependencies:
- `bashpip install -r requirements.txt`
1. Run the application:
- `bashstreamlit run app.py`
- The application will be available at http://localhost:8501

## 📷 Usage
- Upload an image using the file uploader
- Configure options in the sidebar:
- Toggle advanced mode for multiple detailed captions
- Select social media style (Standard, Instagram, Twitter)
- Adjust temperature for creativity control
- View generated captions and copy the ones you like

## ⚙️ Configuration Options
- Multiple Caption Mode: Generate several variations of captions
- Social Media Styling:
- Instagram: Longer captions with emojis, hashtags, and engagement prompts
- Twitter: Shorter, punchier captions with selective hashtags
- Temperature: Control the creativity vs. accuracy tradeoff (0.5-1.5)

## 🌟 Examples
- **Standard Caption**: 
- A golden retriever running through a field of flowers.
- **Instagram Style**:
- 🐶 A golden retriever running through a field of flowers. 🌸
- Tag someone who loves dogs and nature!
- #dogsofinstagram #petsofinstagram #naturelovers
- **Twitter Style**: 
- A golden retriever living its best life in a field of wildflowers 🐶 ✨
- #dogsofinstagram

## 🔍 How It Works
- SnapScribe uses Salesforce's BLIP (Bootstrapping Language-Image Pre-training) model, which combines computer vision and natural language processing to understand image content and generate appropriate descriptions. For social media enhancements, it analyzes the base caption to identify themes and adds relevant hashtags, emojis, and engagement phrases.
- 🛣️ Roadmap
-  Add support for video captioning
-  Implement multilingual captions
-  Add more social media platform styles
 - Create an option to save captioning history
 - Add batch processing for multiple images
   
## 📄 License
- This project is licensed under the MIT License - see the LICENSE file for details.

## Author:
- Made with ❤️ by Melisa Sever
