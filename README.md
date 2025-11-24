# PRAMAN AI - AI Powered Change Detection System

A Streamlit application that uses deep learning to detect changes between images and provides AI-powered insights using GPT-OSS API.

## Features

- **Image Upload & Comparison**: Upload two images to compare for changes
- **Hash-based Ledger**: Track image versions with SHA-256 hashing
- **AI-Powered Analysis**: Uses TensorFlow for change detection
- **Gemini API Integration**: Get detailed insights about detected changes

## Project Structure

```
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── my_change_detection_model.h5   # Trained TensorFlow model
├── ledger.json                     # Image hash ledger
└── README.md                       # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Mohammed-Muhaimin/PRAMAN-AI
cd 
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Dependencies

- **streamlit** - Web application framework
- **tensorflow** - Deep learning framework
- **pillow** - Image processing
- **numpy** - Numerical computing
- **google-generativeai** - gpt oss API integration

## API Configuration

The application uses gpt oss API for AI-powered insights. Make sure your API key is properly configured.

## Notes

- Model file (`my_change_detection_model.h5`) should be pre-trained
- The ledger (`ledger.json`) tracks image hashes for version control
- Ensure proper API key management and security practices

## Author

Mohammed-Muhaimin
Abdulla-Khan
Syed-Abdulla-Nawaz

## License

MIT License
