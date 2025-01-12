# AI CV Screening Tool

## Overview
The AI CV Screening Tool is a powerful application that helps streamline the recruitment process by automatically analyzing and ranking resumes based on job descriptions. It supports both Arabic and English content, making it versatile for international recruitment.

## Features

- **Resume Upload**: Upload multiple PDF or CSV resumes for processing.
- **Job Description Matching**: Enter the job description to match against the uploaded resumes.
- **Language Detection**: Automatically detects whether the resumes and job description are in Arabic or English.
- **Resume Processing**: Processes the uploaded resumes and calculates their similarity to the job description.
- **Similarity Scoring**: Resumes are ranked based on their similarity score to the job description.
- **Evaluation**: Provides evaluation metrics such as precision, recall, and F1 score based on true labels of chosen candidates.


## Technical Stack
- **Frontend**: Streamlit
- **NLP**: NLTK, Sentence Transformers
- **Machine Learning**: scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Document Processing**: docling

## Installation

### Prerequisites
- Python 3.8 or higher


### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-cv-screening-tool.git
cd ai-cv-screening-tool
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Requirements.txt
```
pandas
matplotlib
seaborn
numpy
torch
nltk
transformers
scikit-learn
sentence-transformers
docling
streamlit
plotly
langdetect

```

## Usage

### Starting the Application
```bash
streamlit run app.py
```

## How to Use

### 1. Upload & Process CVs
- Upload multiple PDF resumes or a CSV file containing resumes.
- Paste the job description to match against the resumes.
- Click **Process Resumes** to calculate similarity scores.

### 2. Results & Analysis
- Once resumes are processed, go to **Results & Analysis** to view the ranking of resumes based on their similarity scores to the job description.
- You can visualize the results and analyze the effectiveness of the screening.

### 3. Evaluation
- If you have labeled data (true/false or selected candidates), you can evaluate the precision, recall, and F1 score for the top-ranked resumes.

## How It Works

1. **Text Preprocessing**: The tool preprocesses resumes and job descriptions by tokenizing, removing stopwords, and chunking the text.
2. **Embedding Calculation**: Using the `intfloat/multilingual-e5-large-instruct` SentenceTransformer model for generating embeddings., text embeddings are generated for both resumes and the job description.
3. **Similarity Scoring**: The cosine similarity between the job description embedding and each resume embedding is calculated.
4. **Ranking**: Resumes are ranked based on their similarity scores, with the most relevant resumes appearing at the top.
5. **Evaluation**: If true labels are provided, precision, recall, and F1 scores are calculated to evaluate the quality of the screening process.


## Best Practices for Use

### Resume Format
- Ensure PDFs are text-searchable (not scanned images)
- For CSV files, ensure resume text is in a single column
- Maintain consistent formatting across documents

### Job Descriptions
- Be specific and detailed in job requirements
- Include both technical and soft skills
- Use clear, standard terminology

### Evaluation
- Use a balanced dataset for true label comparison
- Consider both precision and recall metrics
- Review confusion matrix for understanding errors

## Troubleshooting

### Common Issues
1. **PDF Processing Errors**
   - Ensure PDFs are not password-protected
   - Verify PDFs contain searchable text
   - Check file permissions

2. **Language Detection Issues**
   - Ensure text is properly encoded
   - Check for mixed language content
   - Verify text formatting

3. **Performance Issues**
   - Reduce batch size for large datasets
   - Close other resource-intensive applications
   - Check available system memory

### Error Messages
- "No module named 'nltk'": Run `pip install nltk`
- "Failed to process PDF": Check PDF format and permissions
- "Memory error": Reduce batch size or free system memory

## Done

