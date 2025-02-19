import streamlit as st
import pandas as pd
import numpy as np
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from langdetect import detect, DetectorFactory
import json
from datetime import datetime
import tempfile
import os
from docling.document_converter import DocumentConverter
from io import BytesIO
import PyPDF2
# for chat
from typing import Tuple
import requests
from typing import List, Dict, Any


# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

# Page configuration with modern theme
st.set_page_config(
    page_title=" CV Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Update the CSS part of your code with these enhanced styles
st.markdown("""
    <style>
    /* Base font settings */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Header styling */
    h1 {
        color: #2c3e50;
        font-size: 28px !important;
        font-weight: 700;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem;
    }

    h2 {
        color: #34495e;
        font-size: 22px !important;
        font-weight: 600;
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }

    h3 {
        color: #34495e;
        font-size: 18px !important;
        font-weight: 600;
        margin-top: 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Button styling */
    .stButton>button {
        font-size: 14px !important;
        padding: 0.5rem 1rem !important;
        height: auto !important;
    }

    /* Requirements section boxes */
    .requirements-section {
        background-color: white;
        padding: 1rem !important;
        border-radius: 8px;
        margin: 0.75rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .requirements-section h3 {
        font-size: 16px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Dashboard card */
    .dashboard-card {
        background-color: white;
        padding: 1rem !important;
        border-radius: 8px;
        margin: 0.75rem 0;
        border: 1px solid #e9ecef;
    }

    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 1rem !important;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #0066cc;
    }

    .metric-value {
        font-size: 20px !important;
        font-weight: 600;
        color: #0066cc;
        margin-bottom: 0.25rem;
    }

    .metric-label {
        font-size: 12px !important;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Form elements */
    .stTextInput>div>div>input {
        font-size: 14px !important;
    }

    .stTextArea>div>div>textarea {
        font-size: 14px !important;
    }

    .stSelectbox>div>div>div {
        font-size: 14px !important;
    }

    /* Slider labels */
    .stSlider>div>div>div>div {
        font-size: 12px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 14px !important;
        padding: 0.5rem !important;
    }

    /* Table/DataFrame */
    .dataframe {
        font-size: 13px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 14px !important;
        padding: 0.5rem 1rem !important;
    }

    /* Sidebar */
    .css-1d391kg {
        font-size: 14px !important;
    }

    /* Info boxes */
    .stAlert {
        font-size: 14px !important;
        padding: 0.75rem !important;
    }

    /* Custom box for section headers */
    .section-header {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0.5rem 1rem !important;
        margin-bottom: 0.75rem;
        display: inline-block;
        width: auto !important;
    }

    .section-header h3 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 16px !important;
        color: #2c3e50;
    }

     .resume-card {
        background-color: white;
        border: 1px solid #e6e6e6;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    .resume-info {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .resume-name {
        font-size: 14px;
        font-weight: 500;
        color: #2c3e50;
    }

    .language-tag {
        background-color: #e8f0fe;
        color: #0066cc;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
    }

    .score-box {
        background-color: #f8f9fa;
        border: 1px solid #0066cc;
        border-radius: 4px;
        padding: 4px 10px;
        text-align: center;
        min-width: 100px;
    }

    .score-label {
        font-size: 10px;
        color: #666;
        margin-bottom: 2px;
    }

    .score-value {
        font-size: 16px;
        font-weight: 600;
        color: #0066cc;
    }

    .rtl-container {
        direction: rtl;
        text-align: right;
    }

    .ltr-container {
        direction: ltr;
        text-align: left;
    }

    .stTextArea textarea {
        font-family: Arial, sans-serif;
        line-height: 1.4;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "sorted_resumes" not in st.session_state:
        st.session_state.sorted_resumes = None
    if "resumes_data" not in st.session_state:
        st.session_state.resumes_data = pd.DataFrame()
    if "ranking_criteria" not in st.session_state:
        st.session_state.ranking_criteria = {
            "location_weight": 0.0,
            "skills_weight": 0.0,
            "experience_weight": 0.0,
            "education_weight": 0.0,
            "content_weight": 0.0,
            "preferred_location": "",
            "required_skills": [],
            "required_experience": 0,
            "required_degree": "Bachelor",
            "minimum_requirements": {
                "skills_threshold": 0.0,
                "experience_threshold": 0,
                "education_level": "Bachelor"
            }
        }
    if "analysis_timestamp" not in st.session_state:
        st.session_state.analysis_timestamp = None


# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-large-instruct")


model = load_model()

# Language detection setup
DetectorFactory.seed = 0


# Text processing functions
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"


def preprocess_text(text: str) -> Tuple[str, str]:
    if not isinstance(text, str):
        return '', 'en'

    lang = detect_language(text)
    if lang == 'ar':
        processed = preprocess_arabic(text)
    else:
        processed = preprocess_english(text)

    return processed, lang


def preprocess_english(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)


def preprocess_arabic(text: str) -> str:
    text = re.sub(r'[إأٱآا]', 'ا', text)
    text = re.sub(r'[ى]', 'ي', text)
    text = re.sub(r'[ة]', 'ه', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    tokens = word_tokenize(text)
    arabic_stopwords = set(stopwords.words('arabic'))
    tokens = [word for word in tokens if word not in arabic_stopwords]
    return ' '.join(tokens)


def split_text_into_chunks_with_overlap(text, chunk_size=512, overlap_size=10):
    """
    Split text into overlapping chunks for better processing of long documents.

    Args:
        text (str): Input text to be split
        chunk_size (int): Maximum size of each chunk
        overlap_size (int): Number of overlapping tokens between chunks

    Returns:
        list: List of text chunks
    """
    if not isinstance(text, str):
        return []

    tokens = word_tokenize(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size - overlap_size)]
    return [' '.join(chunk) for chunk in chunks]


def extract_text_from_pdf(file: BytesIO) -> str:
    """
    Extract text from a PDF file using either PyPDF2 (for Arabic) or docling (for English).
    Args:
        file (BytesIO): PDF file in BytesIO format
    Returns:
        str: Extracted text in markdown format
    """
    try:
        # First try to detect language from a sample of the PDF
        temp_reader = PyPDF2.PdfReader(file)
        sample_text = ""
        # Try to get text from first few pages for language detection
        for i in range(min(3, len(temp_reader.pages))):
            sample_text += temp_reader.pages[i].extract_text()
        file.seek(0)  # Reset file pointer after sampling

        # Detect language from sample
        try:
            detected_lang = detect(sample_text)
        except:
            detected_lang = 'en'  # Default to English if detection fails

        if detected_lang == 'ar':
            # Use PyPDF2 for Arabic PDFs with enhanced text processing
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []

                for page in pdf_reader.pages:
                    raw_text = page.extract_text()
                    if raw_text:
                        # Process the text line by line
                        lines = raw_text.split('\n')
                        processed_lines = []

                        for line in lines:
                            if line.strip():  # Skip empty lines
                                if any('\u0600' <= c <= '\u06FF' for c in line):
                                    # Process Arabic text
                                    words = line.split()
                                    words.reverse()
                                    processed_line = ' '.join(words)
                                else:
                                    processed_line = line
                                processed_lines.append(processed_line)

                        text_content.append('\n'.join(processed_lines))

                extracted_text = '\n'.join(text_content)
                if not extracted_text.strip():
                    return "No text could be extracted from the Arabic PDF"
                return extracted_text

            except Exception as e:
                st.error(f"Error processing Arabic PDF: {str(e)}")
                return "Error processing Arabic PDF"

        else:
            # Use docling for English PDFs
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    try:
                        # Write BytesIO content to temporary file
                        temp_file.write(file.getvalue())
                        temp_file.flush()

                        # Initialize the document converter
                        converter = DocumentConverter()

                        # Convert the PDF
                        result = converter.convert(temp_file.name)

                        # Get text in markdown format
                        text = result.document.export_to_markdown()
                        return text if text else "No text could be extracted from the English PDF"

                    finally:
                        # Clean up the temporary file
                        try:
                            os.unlink(temp_file.name)
                        except Exception as e:
                            st.warning(f"Warning: Could not delete temporary file: {str(e)}")

            except Exception as e:
                st.error(f"Error processing English PDF: {str(e)}")
                return "Error processing English PDF"

    except Exception as e:
        st.error(f"Error in PDF extraction: {str(e)}")
        return "Error in PDF extraction"
    finally:
        # Always reset file pointer before returning
        file.seek(0)


def parse_cv(uploaded_file: BytesIO) -> Dict:
    url = "https://ai.kayanhr.com/cvparse"
    try:
        files = {'cv': (uploaded_file.name, uploaded_file, 'application/pdf')}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error in CV parsing: {str(e)}")
        return {}


def normalize_degree(degree: str) -> str:
    """Normalize degree string for comparison"""
    # Convert to lowercase and remove extra spaces
    degree = degree.lower().strip()
    # Remove common variations
    replacements = {
        "bachelor's": "bachelor",
        "bachelors": "bachelor",
        "master's": "master",
        "masters": "master",
        "degree": "",
        "'s": "",
        "of": "",
        "  ": " "  # Remove double spaces
    }
    for old, new in replacements.items():
        degree = degree.replace(old, new)
    return degree.strip()


def get_degree_level(degree: str) -> int:
    degree_levels = {
        "diploma": 1,
        "bachelor": 2,
        "bachelors": 2,
        "undergraduate": 2,
        "master": 3,
        "masters": 3,
        "phd": 4,
    }
    return degree_levels.get(normalize_degree(degree), 0)


def calculate_scores(
        resume_text: str,
        parsed_data: Dict,
        job_desc_embedding: np.ndarray,
        ranking_criteria: Dict
) -> Dict[str, float]:
    # Initialize scores
    scores = {
        "content_score": 0.0,
        "location_score": 0.0,
        "skills_score": 0.0,
        "experience_score": 0.0,
        "education_score": 0.0,
        "total_score": 0.0
    }

    # Initialize weights from ranking criteria
    weights = {
        "content": ranking_criteria.get("content_weight", 0.0),
        "location": ranking_criteria.get("location_weight", 0.0),
        "skills": ranking_criteria.get("skills_weight", 0.0),
        "experience": ranking_criteria.get("experience_weight", 0.0),
        "education": ranking_criteria.get("education_weight", 0.0)
    }

    # Track which components have valid scores
    active_weights = {}

    # Content similarity
    try:
        if resume_text.strip() and weights["content"] > 0:
            processed_resume, _ = preprocess_text(resume_text)
            resume_chunks = split_text_into_chunks_with_overlap(processed_resume)
            resume_chunk_embeddings = model.encode(resume_chunks)
            final_resume_embedding = np.mean(resume_chunk_embeddings, axis=0)

            similarity = float(cosine_similarity(
                job_desc_embedding.reshape(1, -1),
                final_resume_embedding.reshape(1, -1)
            )[0][0])

            if similarity > 0:
                scores["content_score"] = similarity
                active_weights["content"] = weights["content"]  # Only add weight if we have a score
    except Exception as e:
        st.warning(f"Error calculating content similarity: {str(e)}")

    # Location matching
    if ranking_criteria["preferred_location"] and weights["location"] > 0:
        cv_locations = [loc.lower() for loc in parsed_data.get("info", {}).get("location", [])]
        if cv_locations:  # Only if we have location data
            preferred_location = ranking_criteria["preferred_location"].lower()
            if any(preferred_location in loc for loc in cv_locations):
                scores["location_score"] = 1.0
                active_weights["location"] = weights["location"]  # Only add weight if we have a match

    # Skills matching
    if weights["skills"] > 0:
        required_skills = set(s.lower() for s in ranking_criteria["required_skills"])
        cv_skills = set(s.lower() for s in parsed_data.get("info", {}).get("Skills", []))
        if required_skills and cv_skills:  # Only if we have both required and CV skills
            matched_skills = len(required_skills.intersection(cv_skills))
            if matched_skills > 0:
                scores["skills_score"] = matched_skills / len(required_skills)
                active_weights["skills"] = weights["skills"]  # Only add weight if we have matches

    # Experience matching
    if weights["experience"] > 0:
        cv_experience = len(parsed_data.get("info", {}).get("position", []))
        required_exp = ranking_criteria["required_experience"]
        if required_exp > 0 and cv_experience > 0:  # Only if we have both required and actual experience
            score = min(cv_experience / required_exp, 1.0)
            if score > 0:
                scores["experience_score"] = score
                active_weights["experience"] = weights["experience"]  # Only add weight if we have valid experience

    # In calculate_scores function:
    if weights["education"] > 0:
        cv_degrees = parsed_data.get("info", {}).get("Degree", [])
        required_degree = ranking_criteria["required_degree"]

        if cv_degrees and required_degree:
            # Print debug information
            print(f"CV Degrees before normalization: {cv_degrees}")
            print(f"Required Degree before normalization: {required_degree}")

            # Normalize degrees
            required_norm = normalize_degree(required_degree)
            cv_degrees_norm = [normalize_degree(deg) for deg in cv_degrees]

            # Print normalized values
            print(f"Normalized CV Degrees: {cv_degrees_norm}")
            print(f"Normalized Required Degree: {required_norm}")

            # Check for match
            has_match = any(required_norm in deg_norm for deg_norm in cv_degrees_norm)
            if has_match:
                scores["education_score"] = 1.0
                active_weights["education"] = weights["education"]
                print(f"Match found! Education score: {scores['education_score']}")
            else:
                print("No match found")

    # Calculate final score using only weights where we have valid scores
    total_active_weight = sum(active_weights.values())

    if total_active_weight > 0:
        weighted_sum = sum(scores[f"{component}_score"] * weight
                           for component, weight in active_weights.items())
        scores["total_score"] = weighted_sum / total_active_weight

        # Store which components were actually used
        scores["components_used"] = list(active_weights.keys())
        scores["weights_used"] = active_weights

    return scores




# Visualization functions
def create_radar_chart(
        candidates_data: List[Dict[str, Any]],
        categories: List[str]
) -> go.Figure:
    fig = go.Figure()

    for candidate in candidates_data:
        fig.add_trace(go.Scatterpolar(
            r=[candidate['scores'][f"{cat.lower()}_score"] for cat in categories],
            theta=categories,
            fill='toself',
            name=candidate['name']
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Candidate Comparison"
    )

    return fig


def create_skills_distribution_chart(filtered_resumes: pd.DataFrame) -> go.Figure:
    all_skills = []
    for _, row in filtered_resumes.iterrows():
        skills = row['parsed_data'].get('info', {}).get('Skills', [])
        all_skills.extend(skills)

    skills_count = pd.Series(all_skills).value_counts().head(10)

    fig = go.Figure(go.Bar(
        x=skills_count.values,
        y=skills_count.index,
        orientation='h',
        marker_color='#0066cc'
    ))

    fig.update_layout(
        title='Top 10 Skills Distribution',
        xaxis_title='Count',
        yaxis_title='Skills',
        height=400,
        showlegend=False,
        plot_bgcolor='white'
    )

    return fig


def create_education_distribution_chart(filtered_resumes: pd.DataFrame) -> go.Figure:
    education_data = []
    for _, row in filtered_resumes.iterrows():
        degrees = row['parsed_data'].get('info', {}).get('Degree', [])
        education_data.extend(degrees)

    edu_count = pd.Series(education_data).value_counts()

    fig = go.Figure(go.Pie(
        labels=edu_count.index,
        values=edu_count.values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3)
    ))

    fig.update_layout(
        title='Education Distribution',
        height=400,
        showlegend=True
    )
    return fig


def create_experience_distribution_chart(filtered_resumes: pd.DataFrame) -> go.Figure:
    experience_data = []
    for _, row in filtered_resumes.iterrows():
        exp = len(row['parsed_data'].get('info', {}).get('position', []))
        experience_data.append(exp)

    fig = go.Figure(go.Histogram(
        x=experience_data,
        nbinsx=10,
        marker_color='#28a745'
    ))

    fig.update_layout(
        title='Experience Distribution',
        xaxis_title='Years of Experience',
        yaxis_title='Count',
        height=400,
        plot_bgcolor='white'
    )

    return fig


def create_simple_dashboard(filtered_resumes: pd.DataFrame):
    """Create a simple, easy-to-understand dashboard with clear visualizations."""

    # Color scheme
    colors = {
        'main': '#0066cc',
        'secondary': '#28a745',
        'accent': '#ffc107',
        'background': '#ffffff'
    }

    # 1. Simple Bar Chart for Top Skills
    def create_skills_bar_chart():
        all_skills = []
        for _, row in filtered_resumes.iterrows():
            skills = row['parsed_data'].get('info', {}).get('Skills', [])
            all_skills.extend(skills)

        skills_count = pd.Series(all_skills).value_counts().head(10)

        fig = go.Figure(go.Bar(
            x=skills_count.values,
            y=skills_count.index,
            orientation='h',
            marker_color=colors['main']
        ))

        fig.update_layout(
            title='Most Common Skills',
            xaxis_title='Number of Candidates',
            yaxis_title='Skills',
            plot_bgcolor=colors['background'],
            height=400,
            showlegend=False
        )

        return fig

    # 2. Simple Pie Chart for Education
    def create_education_pie():
        education_data = []
        for _, row in filtered_resumes.iterrows():
            degrees = row['parsed_data'].get('info', {}).get('Degree', [])
            education_data.extend(degrees)

        edu_count = pd.Series(education_data).value_counts()

        fig = go.Figure(go.Pie(
            labels=edu_count.index,
            values=edu_count.values,
            marker=dict(colors=[colors['main'], colors['secondary'], colors['accent']])
        ))

        fig.update_layout(
            title='Education Distribution',
            height=400
        )

        return fig

    # 3. Simple Bar Chart for Experience
    def create_experience_bar():
        exp_ranges = ['0-2 years', '3-5 years', '6-10 years', '10+ years']
        experience_counts = [0, 0, 0, 0]

        for _, row in filtered_resumes.iterrows():
            exp = len(row['parsed_data'].get('info', {}).get('position', []))
            if exp <= 2:
                experience_counts[0] += 1
            elif exp <= 5:
                experience_counts[1] += 1
            elif exp <= 10:
                experience_counts[2] += 1
            else:
                experience_counts[3] += 1

        fig = go.Figure(go.Bar(
            x=exp_ranges,
            y=experience_counts,
            marker_color=colors['secondary']
        ))
        fig.update_layout(
            title='Experience Distribution',
            xaxis_title='Years of Experience',
            yaxis_title='Number of Candidates',
            plot_bgcolor=colors['background'],
            height=400,
            showlegend=False
        )

        return fig

    # 4. Simple Bar Chart for Locations
    def create_location_bar():
        locations = []
        for _, row in filtered_resumes.iterrows():
            locs = row['parsed_data'].get('info', {}).get('location', [])
            locations.extend(locs)

        loc_count = pd.Series(locations).value_counts().head(8)

        fig = go.Figure(go.Bar(
            x=loc_count.index,
            y=loc_count.values,
            marker_color=colors['accent']
        ))

        fig.update_layout(
            title='Top Locations',
            xaxis_title='Location',
            yaxis_title='Number of Candidates',
            plot_bgcolor=colors['background'],
            height=400,
            showlegend=False
        )

        return fig

    # Display the simplified dashboard
    st.header("📊 Dashboard")

    # Simple Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Candidates",
            len(filtered_resumes)
        )

    with col2:
        avg_score = filtered_resumes['total_score'].mean()
        st.metric(
            "Average Match",
            f"{avg_score:.0%}"
        )

    with col3:
        top_score = filtered_resumes['total_score'].max()
        st.metric(
            "Best Match",
            f"{top_score:.0%}"
        )

    with col4:
        qualified = len(filtered_resumes[filtered_resumes['total_score'] >= 0.7])
        st.metric(
            "Qualified Candidates",
            qualified
        )

    # First Row - Skills and Education
    st.subheader("Skills and Education")
    col1, col2 = st.columns(2)

    with col1:
        skills_chart = create_skills_bar_chart()
        st.plotly_chart(skills_chart, use_container_width=True)

    with col2:
        education_chart = create_education_pie()
        st.plotly_chart(education_chart, use_container_width=True)

    # Second Row - Experience and Location
    st.subheader("Experience and Location")
    col1, col2 = st.columns(2)

    with col1:
        experience_chart = create_experience_bar()
        st.plotly_chart(experience_chart, use_container_width=True)

    with col2:
        location_chart = create_location_bar()
        st.plotly_chart(location_chart, use_container_width=True)

    # Simple Summary Table
    st.subheader("Top Candidates")
    top_candidates = filtered_resumes.head(5)[['filename', 'total_score']]
    top_candidates['total_score'] = top_candidates['total_score'].apply(lambda x: f"{x:.0%}")

    st.dataframe(
        top_candidates.rename(columns={
            'filename': 'Resume',
            'total_score': 'Match Score'
        }),
        use_container_width=True
    )

def display_detailed_scores(scores: Dict[str, float], parsed_data: Dict):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Detailed Scores")
        for criterion, score in scores.items():
            if criterion != "total_score":
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{criterion.replace('_score', ' Match').title()}</div>
                        <div class="metric-value">{score:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col2:
        st.subheader("📑 Parsed Information")
        info = parsed_data.get("info", {})

        st.markdown("**🎓 Education:**")
        for degree in info.get("Degree", []):
            st.markdown(f"- {degree}")

        st.markdown("**💼 Experience:**")
        st.markdown(f"- {len(info.get('position', []))} positions")

        st.markdown("**📍 Location:**")
        for location in info.get("location", []):
            st.markdown(f"- {location}")

        st.markdown("**🛠 Skills:**")
        skills_text = ", ".join(info.get("Skills", []))
        st.markdown(skills_text if skills_text else "No skills found")


import pandas as pd
import ollama
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re


class CVDatabaseChat:
    def __init__(self):
        self.cv_data = pd.DataFrame()

    def add_cv(self, cv_info: Dict):
        """Add new CV to database"""
        # Convert the skills list from string back to actual list if it's a string
        if isinstance(cv_info.get('skills'), str):
            try:
                cv_info['skills'] = eval(cv_info['skills'])
            except:
                cv_info['skills'] = []

        # Create a new DataFrame with the single CV
        new_cv = pd.DataFrame([cv_info])

        # Concatenate with existing data
        self.cv_data = pd.concat([self.cv_data, new_cv], ignore_index=True)
        logging.info(f"Added new CV: {cv_info.get('filename', 'Unknown')}")

    def format_context(self) -> str:
        """Format CV database information for model context"""
        if self.cv_data.empty:
            return "No CVs in database."

        context = "CV Database Information:\n\n"
        for _, row in self.cv_data.iterrows():
            context += f"File: {row['filename']}\n"
            # Handle skills appropriately based on its type
            skills = row['skills']
            if isinstance(skills, str):
                try:
                    skills = eval(skills)
                except:
                    skills = []
            skills_str = ', '.join(skills) if isinstance(skills, list) else str(skills)

            context += f"Skills: {skills_str}\n"
            context += f"Location: {row['location']}\n"
            context += f"Experience: {row['experience']} years\n"
            context += f"Education: {row['education']}\n"
            context += f"Score: {row['total_score']:.0%}\n"
            context += f"Resume Text: {row.get('resume_text', 'Not available')[:500]}...\n\n"
        return context

    def process_query(self, query: str) -> str:
        """Process user query about the CVs"""
        try:
            # Prepare context and system message
            context = self.format_context()
            system_message = f"""You are a CV analysis assistant. Answer questions based on the following CV database:

{context}

Important guidelines:
1. Only use information present in the database
2. Show exact score values as percentages
3. When discussing skills, list them exactly as shown
4. If information is not in the database, say so
5. Keep answers concise but informative
6. For experience, use exact years from the database
7. Include specific details from resumes when relevant"""

            # Prepare messages for the chat
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            # Get response from model
            response = ollama.chat(
                model='deepseek-r1',
                messages=messages,
                stream=False
            )

            # Get the response content
            description = response['message']['content']

            # Remove thinking parts using regex
            description = re.sub(r'<think>.*?</think>', '', description, flags=re.DOTALL)

            # Clean up any extra whitespace that might be left
            description = re.sub(r'\n\s*\n', '\n\n', description.strip())

            return description

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"


# Function to save CV to database
def save_cv_to_database(cv_info: Dict):
    """Save processed CV information to database"""
    if 'cv_chat' not in st.session_state:
        st.session_state.cv_chat = CVDatabaseChat()
    st.session_state.cv_chat.add_cv(cv_info)


# Modified version of the chat tab code
def display_chat_tab():
    st.subheader("💬 Chat with CV Assistant")

    # Initialize chat if needed
    if 'cv_chat' not in st.session_state:
        st.session_state.cv_chat = CVDatabaseChat()

    # Show database status
    num_cvs = len(st.session_state.cv_chat.cv_data)
    st.info(f"Database contains {num_cvs} CVs")

    # Chat interface
    if prompt := st.chat_input("Ask about the CVs..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.cv_chat.process_query(prompt)
            st.markdown(response)

def main():

    init_session_state()

    st.title("🎯  CV Analyzer")

    # Sidebar navigation
    with st.sidebar:
        page = st.radio("📍 Navigation", ["Upload & Process", "Results", "Settings"])

        with st.expander("ℹ️ About"):
            st.markdown("""
            This advanced CV screening tool helps you:
            - Process multiple resumes efficiently
            - Match candidates to job requirements
            - Get detailed analytics and insights
            - Compare candidates side by side
            - Export analysis results
            """)

    if page == "Upload & Process":
        st.header("📤 Upload & Process")

        # File upload section
        upload_col1, upload_col2 = st.columns([1, 1])

        with upload_col1:
            st.subheader("📄 Upload Resumes")
            uploaded_files = st.file_uploader(
                "Drop PDF files here",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload multiple PDF resumes for analysis"
            )

        with upload_col2:
            st.subheader("📝 Job Description")
            job_description = st.text_area(
                "Enter the job description",
                height=200,
                help="Paste the complete job description here"
            )

        # Requirements and Weights Section
        st.header("⚖️ Requirements and Weights")

        with st.container():
            st.markdown("""
                <div class="section-header">
                    <h3>📍 Location Requirements</h3>
                </div>
            """, unsafe_allow_html=True)

            loc_col1, loc_col2 = st.columns([2, 1])
            with loc_col1:
                st.session_state.ranking_criteria["preferred_location"] = st.text_input(
                    "Preferred Location",
                    help="Enter the preferred work location"
                )
            with loc_col2:
                st.session_state.ranking_criteria["location_weight"] = st.slider(
                    "Location Weight",
                    0.0, 1.0,
                    help="Set the importance of location match"
                )

        with st.container():
            st.markdown("""
                <div class="requirements-section">
                    <h3>🛠️ Skills Requirements</h3>
                </div>
            """, unsafe_allow_html=True)

            skills_col1, skills_col2 = st.columns([2, 1])
            with skills_col1:
                skills_input = st.text_area(
                    "Required Skills (one per line)",
                    help="Enter each required skill on a new line"
                )
                st.session_state.ranking_criteria["required_skills"] = [
                    s.strip() for s in skills_input.split('\n') if s.strip()
                ]
            with skills_col2:
                st.session_state.ranking_criteria["skills_weight"] = st.slider(
                    "Skills Weight",
                    0.0, 1.0,
                    help="Set the importance of skills match"
                )

        with st.container():
            st.markdown("""
                <div class="section-header">
                    <h3>💼 Experience Requirements</h3>
                </div>
            """, unsafe_allow_html=True)

            exp_col1, exp_col2 = st.columns([2, 1])
            with exp_col1:
                st.session_state.ranking_criteria["required_experience"] = st.number_input(
                    "Minimum Years of Experience",
                    min_value=0,
                    max_value=20,

                    help="Enter the minimum required years of experience"
                )
            with exp_col2:
                st.session_state.ranking_criteria["experience_weight"] = st.slider(
                    "Experience Weight",
                    0.0, 1.0,
                    help="Set the importance of experience match"
                )

        with st.container():
            st.markdown("""
                <div class="section-header">
                    <h3>🎓 Education Requirements</h3>
                </div>
            """, unsafe_allow_html=True)

            edu_col1, edu_col2 = st.columns([2, 1])
            with edu_col1:
                st.session_state.ranking_criteria["required_degree"] = st.selectbox(
                    "Minimum Education Level",
                    ["Diploma", "Bachelor", "Master", "PhD"],
                    help="Select the minimum required education level"
                )
            with edu_col2:
                st.session_state.ranking_criteria["education_weight"] = st.slider(
                    "Education Weight",
                    0.0, 1.0,
                    help="Set the importance of education match"
                )

        with st.container():
            st.markdown("""
                <div class="requirements-section">
                    <h3>📊 Content Matching</h3>
                </div>
            """, unsafe_allow_html=True)

            content_col1, content_col2 = st.columns([2, 1])
            with content_col1:
                st.info("Measures how well the resume content matches the job description.")
            with content_col2:
                st.session_state.ranking_criteria["content_weight"] = st.slider(
                    "Content Match Weight",
                    0.0, 1.0,
                    help="Set the importance of content similarity"
                )

        # Process Button
        if st.button("🚀 Process Resumes", type="primary", use_container_width=True):
            if uploaded_files and job_description:
                with st.spinner("📊 Processing resumes..."):
                    # Process job description with chunking
                    processed_job_desc, _ = preprocess_text(job_description)
                    job_desc_chunks = split_text_into_chunks_with_overlap(processed_job_desc)
                    job_desc_chunk_embeddings = model.encode(job_desc_chunks)
                    job_desc_embedding = np.mean(job_desc_chunk_embeddings, axis=0)

                    all_scores = []
                    all_texts = []
                    all_parsed_data = []

                    progress_bar = st.progress(0)
                    total_files = len(uploaded_files)

                    for idx, file in enumerate(uploaded_files):
                        try:
                            # Extract and process resume text
                            resume_text = extract_text_from_pdf(file)
                            if not resume_text or resume_text.startswith("Error"):
                                st.error(f"Could not extract text from {file.name}")
                                continue

                            # Parse CV
                            file.seek(0)
                            parsed_data = parse_cv(file)
                            if not parsed_data:
                                st.error(f"Could not parse {file.name}")
                                continue

                            # Calculate scores with chunking
                            scores = calculate_scores(
                                resume_text,
                                parsed_data,
                                job_desc_embedding,
                                st.session_state.ranking_criteria
                            )

                            all_scores.append(scores)
                            all_texts.append(resume_text)
                            all_parsed_data.append(parsed_data)

                            cv_info = {
                                "filename": file.name,
                                "skills": str(parsed_data.get("info", {}).get("Skills", [])),
                                "location": ", ".join(parsed_data.get("info", {}).get("location", [])),
                                "experience": len(parsed_data.get("info", {}).get("position", [])),
                                "education": ", ".join(parsed_data.get("info", {}).get("Degree", [])),
                                "total_score": scores["total_score"],
                                "resume_text": resume_text
                            }
                            save_cv_to_database(cv_info)

                            progress_bar.progress((idx + 1) / total_files)

                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                            continue

                    if all_scores:  # Only create DataFrame if we have scores
                        # Create results DataFrame
                        results_df = pd.DataFrame({
                            "filename": [f.name for f in uploaded_files],
                            "resume_text": all_texts,
                            "parsed_data": all_parsed_data,
                            **{k: [s[k] for s in all_scores] for k in all_scores[0].keys()}
                        })

                        # Sort resumes by total score
                        st.session_state.sorted_resumes = results_df.sort_values(
                            by="total_score",
                            ascending=False
                        ).reset_index(drop=True)
                        # Save to chat system database
                        cv_info = {
                            "filename": file.name,
                            "skills": str(parsed_data.get("info", {}).get("Skills", [])),
                            "location": ", ".join(parsed_data.get("info", {}).get("location", [])),
                            "experience": len(parsed_data.get("info", {}).get("position", [])),
                            "education": ", ".join(parsed_data.get("info", {}).get("Degree", [])),
                            "total_score": scores["total_score"],
                            "resume_text": resume_text
                        }
                        save_cv_to_database(cv_info)
                        st.session_state.processed = True
                        st.session_state.analysis_timestamp = datetime.now()

                        st.success("✅ Processing complete!")

                        # Display ranked resumes
                        st.markdown("### 📑 Ranked Resumes")

                        for idx, row in st.session_state.sorted_resumes.iterrows():
                            # Detect language
                            detected_lang = detect_language(row['resume_text'])
                            language = 'Arabic' if detected_lang == 'ar' else 'English'

                            # Display resume card
                            st.markdown(f"""
                                <div class="resume-card">
                                    <div class="resume-info">
                                        <span class="resume-name">📄 Resume {idx + 1}: {row['filename']}</span>
                                        <span class="language-tag">{language}</span>
                                    </div>
                                    <div class="score-box">
                                        <div class="score-label">Match Score</div>
                                        <div class="score-value">{row['total_score']:.1%}</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            # Display details in expander
                            with st.expander("Show Details"):
                                display_detailed_scores(
                                    {k: row[k] for k in row.keys() if k.endswith('_score')},
                                    row['parsed_data']
                                )

                                # Resume content with RTL support
                                st.markdown("### 📝 Resume Content")

                                # Add RTL/LTR container
                                direction_class = "rtl-container" if detected_lang == 'ar' else "ltr-container"
                                st.markdown(f'<div class="{direction_class}">', unsafe_allow_html=True)

                                # Display text area
                                st.text_area(
                                    "",
                                    row['resume_text'],
                                    height=300,
                                    key=f"resume_text_{idx}"
                                )

                                # Close container
                                st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No resumes were successfully processed.")
            else:
                st.error("Please upload resumes and provide a job description.")

    elif page == "Results":
        if not st.session_state.processed or st.session_state.sorted_resumes is None:
            st.info(" Please process some resumes first.")
            return

        # Results Page
        st.header("📊 Analysis Results")

        if st.session_state.analysis_timestamp:
            st.markdown(f"*Last analyzed: {st.session_state.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}*")

        # Create tabs for different views
        tabs = st.tabs(["📈 Dashboard", "📋 Detailed Results", "🔄 Comparison", "💬 Chat"])

        with tabs[0]:
            create_simple_dashboard(st.session_state.sorted_resumes)

        with tabs[1]:
            # Filtering controls section
            st.markdown("""
                <div class="dashboard-card">
                    <h3>🔍 Filter Results</h3>
                </div>
            """, unsafe_allow_html=True)

            # Create two rows of filters with 3 columns each
            row1_cols = st.columns(3)
            row2_cols = st.columns(3)

            # First row of filters
            with row1_cols[0]:
                min_score = st.slider("Minimum Total Score", 0.0, 1.0, 0.0, 0.05)
            with row1_cols[1]:
                min_exp = st.slider("Minimum Experience Score", 0.0, 1.0, 0.0, 0.05)
            with row1_cols[2]:
                min_edu = st.slider("Minimum Education Score", 0.0, 1.0, 0.0, 0.05)

            # Second row of filters
            with row2_cols[0]:
                min_loc = st.slider("Minimum Location Score", 0.0, 1.0, 0.0, 0.05)
            with row2_cols[1]:
                min_skills = st.slider("Minimum Skills Score", 0.0, 1.0, 0.0, 0.05)
            with row2_cols[2]:
                # You could add another filter here if needed
                st.markdown("")  # Empty space for layout balance

            # Apply filters to the dataframe
            filtered_resumes = st.session_state.sorted_resumes[
                (st.session_state.sorted_resumes['total_score'] >= min_score) &
                (st.session_state.sorted_resumes['experience_score'] >= min_exp) &
                (st.session_state.sorted_resumes['education_score'] >= min_edu) &
                (st.session_state.sorted_resumes['location_score'] >= min_loc) &
                (st.session_state.sorted_resumes['skills_score'] >= min_skills)
                # Note: Changed from skill_score to skills_score
                ]

            # Display filtered candidates
            st.subheader(f"📄 Candidate Details ({len(filtered_resumes)} matches)")

            # Add a summary of active filters
            active_filters = []
            if min_score > 0: active_filters.append(f"Total Score ≥ {min_score:.0%}")
            if min_exp > 0: active_filters.append(f"Experience Score ≥ {min_exp:.0%}")
            if min_edu > 0: active_filters.append(f"Education Score ≥ {min_edu:.0%}")
            if min_loc > 0: active_filters.append(f"Location Score ≥ {min_loc:.0%}")
            if min_skills > 0: active_filters.append(f"Skills Score ≥ {min_skills:.0%}")

            if active_filters:
                st.markdown("**Active Filters:**")
                for filter_text in active_filters:
                    st.markdown(f"- {filter_text}")

            # Display filtered candidates
            for idx, row in filtered_resumes.iterrows():
                with st.expander(f"📄 Resume {idx + 1}: {row['filename']} - Match: {row['total_score']:.1%}"):
                    display_detailed_scores(
                        {k: row[k] for k in row.keys() if k.endswith('_score')},
                        row['parsed_data']
                    )

                    # Resume content
                    st.markdown("### 📝 Resume Content")
                    st.text_area(
                        "",
                        row['resume_text'],
                        height=300,
                        key=f"resume_text_{idx}"
                    )

        with tabs[2]:
            if len(st.session_state.sorted_resumes) >= 2:
                st.subheader("👥 Candidate Comparison")

                col1, col2 = st.columns(2)
                with col1:
                    candidate1 = st.selectbox(
                        "Select First Candidate",
                        options=st.session_state.sorted_resumes['filename'].tolist(),
                        key="candidate1"
                    )

                with col2:
                    candidate2 = st.selectbox(
                        "Select Second Candidate",
                        options=st.session_state.sorted_resumes['filename'].tolist(),
                        key="candidate2"
                    )

                if candidate1 and candidate2:
                    cand1_data = st.session_state.sorted_resumes[
                        st.session_state.sorted_resumes['filename'] == candidate1
                        ].iloc[0]
                    cand2_data = st.session_state.sorted_resumes[
                        st.session_state.sorted_resumes['filename'] == candidate2
                        ].iloc[0]

                    # Create radar chart
                    categories = ['Content', 'Location', 'Skills', 'Experience', 'Education']
                    candidates_data = [
                        {'name': candidate1, 'scores': cand1_data},
                        {'name': candidate2, 'scores': cand2_data}
                    ]

                    radar_chart = create_radar_chart(candidates_data, categories)
                    st.plotly_chart(radar_chart, use_container_width=True)

                    # Detailed comparison
                    st.markdown("""
                        <div class="dashboard-card">
                            <h4>📊 Score Comparison</h4>
                        </div>
                    """, unsafe_allow_html=True)

                    comparison_data = {
                        'Metric': ['Total Score'] + categories,
                        candidate1: [cand1_data['total_score']] +
                                    [cand1_data[f"{cat.lower()}_score"] for cat in categories],
                        candidate2: [cand2_data['total_score']] +
                                    [cand2_data[f"{cat.lower()}_score"] for cat in categories]
                    }

                    # Replace the comparison table section with this code
                    comparison_data = {
                        'Metric': ['Total Score'] + categories,
                        candidate1: [cand1_data['total_score']] +
                                    [cand1_data[f"{cat.lower()}_score"] for cat in categories],
                        candidate2: [cand2_data['total_score']] +
                                    [cand2_data[f"{cat.lower()}_score"] for cat in categories]
                    }

                    comparison_df = pd.DataFrame(comparison_data)

                    # Convert to numeric and then format as percentage
                    for col in [candidate1, candidate2]:
                        # First ensure values are numeric
                        comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
                        # Then format as percentage
                        comparison_df[col] = comparison_df[col].map('{:.1%}'.format)

                    # Display the formatted dataframe
                    st.markdown("### 📊 Comparison Table")
                    st.dataframe(
                        comparison_df.style.set_properties(**{
                            'background-color': '#f8f9fa',
                            'font-size': '14px',
                            'padding': '10px'
                        }),
                        use_container_width=True
                    )

                    # Skills comparison
                    st.markdown("""
                        <div class="dashboard-card">
                            <h4>🛠️ Skills Analysis</h4>
                        </div>
                    """, unsafe_allow_html=True)

                    skills1 = set(cand1_data['parsed_data'].get('info', {}).get('Skills', []))
                    skills2 = set(cand2_data['parsed_data'].get('info', {}).get('Skills', []))

                    common_skills = skills1.intersection(skills2)
                    unique_skills1 = skills1 - skills2
                    unique_skills2 = skills2 - skills1

                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown(f"**🤝 Common Skills ({len(common_skills)})**")
                        for skill in sorted(common_skills):
                            st.markdown(f"- {skill}")

                    with cols[1]:
                        st.markdown(f"**1️⃣ Unique to {candidate1} ({len(unique_skills1)})**")
                        for skill in sorted(unique_skills1):
                            st.markdown(f"- {skill}")

                    with cols[2]:
                        st.markdown(f"**2️⃣ Unique to {candidate2} ({len(unique_skills2)})**")
                        for skill in sorted(unique_skills2):
                            st.markdown(f"- {skill}")
            else:
                st.info(" Need at least 2 candidates for comparison.")
        # Add this in the tabs section
        # Add this to your tabs section in the main function
        with tabs[3]:  # Chat tab
            if not st.session_state.processed:
                st.info("Please process some resumes first to enable chat.")
            else:
                display_chat_tab()


    elif page == "Settings":
        st.header("⚙️ Settings")

        # Model Settings
        with st.container():
            st.markdown("""
                <div class="dashboard-card">
                    <h3>🤖 Model Configuration</h3>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("**Current Model:** intfloat/multilingual-e5-large-instruct")
            st.info("This model supports multilingual text processing and semantic analysis.")

            model_cols = st.columns(2)
            with model_cols[0]:
                st.metric("Model Type", "Transformer")
            with model_cols[1]:
                st.metric("Languages", "Multilingual")

        # Parser Settings
        with st.container():
            st.markdown("""
                <div class="dashboard-card">
                    <h3>🔍 Parser Configuration</h3>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("**API Endpoint:** https://ai.kayanhr.com/cvparse")
            st.info("Advanced resume parser for extracting structured information.")

            parser_cols = st.columns(2)
            with parser_cols[0]:
                st.metric("Parser Type", "AI-Powered")
            with parser_cols[1]:
                st.metric("Output Format", "JSON")

        # Export Settings
        with st.container():
            st.markdown("""
                <div class="dashboard-card">
                    <h3>📤 Export Options</h3>
                </div>
            """, unsafe_allow_html=True)

            if st.session_state.processed:
                export_cols = st.columns(2)
                with export_cols[0]:
                    if st.download_button(
                            label="📊 Download Analysis (CSV)",
                            data=st.session_state.sorted_resumes.to_csv(index=False).encode('utf-8'),
                            file_name='cv_analysis_results.csv',
                            mime='text/csv'
                    ):
                        st.success("✅ Download started!")

                with export_cols[1]:
                    # Export detailed report
                    report_data = {
                        "analysis_date": st.session_state.analysis_timestamp,
                        "total_candidates": len(st.session_state.sorted_resumes),
                        "average_score": st.session_state.sorted_resumes['total_score'].mean(),
                        "top_score": st.session_state.sorted_resumes['total_score'].max(),
                        "ranking_criteria": st.session_state.ranking_criteria
                    }

                    if st.download_button(
                            label="📑 Download Detailed Report (JSON)",
                            data=json.dumps(report_data, default=str, indent=2),
                            file_name='cv_analysis_report.json',
                            mime='application/json'
                    ):
                        st.success("✅ Report downloaded!")
            else:
                st.info("👆 Process some resumes to enable export options.")

        # Documentation
        with st.container():
            st.markdown("""
                <div class="dashboard-card">
                    <h3>📚 Documentation</h3>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                ### 📊 Scoring System

                The final score is calculated as a weighted average of multiple components:

                1. **Content Matching **
                   - Semantic similarity between resume and job description
                   - Uses advanced NLP techniques

                2. **Skills Matching **
                   - Direct comparison of required vs. candidate skills
                   - Considers skill relevance and expertise level

                3. **Location Matching **
                   - Geographical alignment with job location
                   - Considers remote work preferences

                4. **Experience Matching **
                   - Years of relevant experience
                   - Role similarity and progression

                5. **Education Matching **
                   - Degree level alignment
                   - Field of study relevance

                Each component is normalized to a 0-1 scale and weighted according to user preferences.
                Weights can be adjusted in the Upload & Process section.
                """)


if __name__ == "__main__":
    main()