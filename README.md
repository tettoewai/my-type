# MyType

A Streamlit web application that uses Computer Vision to analyze your photo and recommend personalized hairstyles, makeup palettes, and outfits based on your detected face shape, skin undertone, and body shape.

## Features

- **Face Shape Detection** - Analyzes 468 facial landmarks to identify your face shape (Oval, Round, Square, Heart, Oblong)
- **Skin Undertone Analysis** - Uses CIELAB color space to determine Warm, Cool, or Neutral undertones
- **Body Shape Detection** - Analyzes body proportions to classify body shape (Hourglass, Triangle, Inverted Triangle, Rectangle, Apple)
- **Personalized Recommendations** - Rule-based engine provides tailored hairstyle, makeup, and outfit suggestions
- **Interactive Visualizations** - Radar charts and bar graphs powered by Plotly
- **Privacy-First** - All processing runs locally on your machine

## Tech Stack

| Technology | Purpose                          |
| ---------- | -------------------------------- |
| Streamlit  | Web application framework        |
| MediaPipe  | Face and pose landmark detection |
| OpenCV     | Image processing                 |
| Plotly     | Interactive data visualizations  |
| Pandas     | Data manipulation                |
| Pillow     | Image handling                   |

## Prerequisites

- Python 3.8 or higher
- Webcam or photo for analysis

## Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

### Steps:

1. Upload a clear, front-facing photo with good lighting
2. Wait for the analysis to complete
3. View your personalized style recommendations

## Project Structure

```
MyType/
├── app.py                          # Main Streamlit application
├── modules/
│   ├── __init__.py
│   ├── face_analyzer.py            # Face shape detection
│   ├── skin_analyzer.py            # Skin undertone analysis
│   ├── body_analyzer.py            # Body shape detection
│   └── recommender.py              # Style recommendation engine
├── utils/                          # Utility functions
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## How It Works

1. **Face Analysis** - MediaPipe FaceMesh detects 468 facial landmarks, calculates face proportions, and determines face shape using fuzzy logic confidence scoring

2. **Skin Analysis** - Extracts cheek area using landmark masks, converts to CIELAB color space, and classifies undertone based on color channel analysis

3. **Body Analysis** - MediaPipe Pose detects 33 body landmarks, measures shoulder-to-hip ratio, and classifies body shape

4. **Recommendations** - Combines all analysis results through a rule-based engine to generate personalized style advice

## Limitations

- Body shape detection is simplified and works best with clear, upright, full-body photos
- Accuracy may vary with poor lighting, angles, or obstructions
- Initial MediaPipe model download requires internet connection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face and pose detection models
- [Streamlit](https://streamlit.io/) for the web framework
- [Plotly](https://plotly.com/) for interactive visualizations
