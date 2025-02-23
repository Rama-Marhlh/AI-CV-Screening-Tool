# CV Analyzer

A powerful CV/Resume analysis tool built with Streamlit that helps recruiters and hiring managers efficiently process, analyze, and compare multiple resumes against job requirements.

## 🌟 Features

- **Multi-Resume Processing**: Upload and analyze multiple PDF resumes simultaneously
- **Multilingual Support**: Process resumes in both English and Arabic
- **Intelligent Matching**: Compare resumes against job descriptions using advanced NLP
- **Customizable Scoring**: Adjust weights for different criteria:
  - Content matching
  - Skills alignment
  - Location preferences
  - Experience requirements
  - Education qualifications
- **Interactive Dashboard**: Visualize candidate distributions and comparisons
- **Detailed Analysis**: Get comprehensive insights for each candidate
- **Chat Interface**: Query your resume database using natural language
- **Export Capabilities**: Download analysis results in CSV and JSON formats

## 🗋l Prerequisites

- Python 3.8+
- PDF processing capabilities
- Internet connection for API access

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cv-analyzer.git
cd cv-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

## 💻 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Navigate to the displayed local URL (typically http://localhost:8501)

3. Use the application:
   - Upload resumes (PDF format)
   - Enter job description
   - Set matching criteria and weights
   - Process and analyze resumes
   - View results and comparisons
   - Export analysis results

## 📊 Analysis Components

### Resume Processing
- PDF text extraction
- Language detection
- Structure parsing
- Data normalization

### Scoring System
- Content similarity using BERT embeddings
- Skills matching
- Location alignment
- Experience evaluation
- Education qualification matching

### Visualization
- Interactive dashboards
- Comparison charts
- Distribution graphs
- Detailed candidate profiles

## 🔧 Configuration

The application supports various configuration options:

- Model settings:
  - Using multilingual-e5-large-instruct model
  - Configurable chunk sizes for text processing
  
- Parser settings:
  - API endpoint configuration
  - Response handling preferences

- Export settings:
  - CSV export options
  - JSON report configuration

## 🤝 API Integration

The system integrates with an external CV parsing API:
- Endpoint: https://ai.kayanhr.com/cvparse
- Authentication: Required
- Format: JSON
- API for configuration: Allows dynamic adjustment of parsing and scoring parameters
- **Production API**: A Flask-based API is required for production deployment, enabling scalable and efficient processing

## 💬 Chat System

The application includes a chat interface that enables querying the resume database using natural language. We use the `deepseek-r1` model for chat responses:
```python
# Get response from model
response = ollama.chat(
    model='deepseek-r1',
    messages=messages,
    stream=False
)
```
To use `deepseek-r1`, install the Ollama package and ensure it's properly configured:
```bash
pip install ollama
```

## 🌍 Using multilingual-e5-large-instruct Model

This model is used for multilingual processing and resume matching. To download and use it:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
```
Install the required package:
```bash
pip install sentence-transformers
```

## 📱 Responsive Design

The interface adapts to different screen sizes:
- Desktop: Full feature set
- Tablet: Optimized layouts
- Mobile: Essential features

## 🛠 Troubleshooting

Common issues and solutions:

1. PDF Extraction Fails:
   - Ensure PDF is not password protected
   - Check for proper UTF-8 encoding
   - Verify PDF is not corrupted

2. API Connection Issues:
   - Check internet connectivity
   - Verify API endpoint status
   - Ensure proper authentication

3. Performance Issues:
   - Reduce batch size
   - Clear browser cache
   - Restart application

## 📚 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request



