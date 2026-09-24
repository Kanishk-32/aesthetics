# AI Photo Coach

**Real-Time Intelligent Photography Assistant**

An AI-powered computer vision system that analyzes facial pose, lighting, composition, sharpness, and framing to provide real-time photography guidance and rank captured images based on configurable photo-quality criteria.

## Features

- **Photo Analysis**: Upload → analyze → feedback
- **Photo Ranking**: Compare multiple photos, identify the strongest shot
- **Aesthetic Intelligence**: Composition, color, lighting style, background analysis
- **Live AI Camera Assistant**: Real-time guidance for better photos
- **Personalization**: Learns your best angles over time

## Project Structure

```
ASTHETICS/
├── src/
│   ├── cv/           # Computer vision modules
│   ├── models/       # Neural network models
│   └── utils/        # Helper functions
├── app/              # Streamlit UI
├── configs/          # Configuration files
├── tests/            # Unit tests
├── data/             # Local data (gitignored)
├── checkpoints/      # Model weights (gitignored)
└── .github/          # CI/CD workflows
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app/main.py
```

## Development Pipeline

See `ai-photo-coach-pipeline.txt` for the complete development roadmap.

## License

MIT
