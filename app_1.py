import nltk
import os

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Call the download function
download_nltk_data()

# Now import word_tokenize after ensuring data is downloaded
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from docling.document_converter import DocumentConverter
from preprocess import preprocess_text
import tempfile
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI CV Screening Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .upload-text {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 1em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if "processed" not in st.session_state:
    st.session_state.processed = False
if "sorted_resumes" not in st.session_state:
    st.session_state.sorted_resumes = None
if "resumes_data" not in st.session_state:
    st.session_state.resumes_data = pd.DataFrame()
if "csv_columns" not in st.session_state:
    st.session_state.csv_columns = []
if "current_view" not in st.session_state:
    st.session_state.current_view = "upload"


# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-large-instruct")


model = load_model()


# Function definitions (keeping the same as before)
def split_text_into_chunks_with_overlap(text, chunk_size=512, overlap_size=10):
    tokens = word_tokenize(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size - overlap_size)]
    return [' '.join(chunk) for chunk in chunks]


def calculate_similarity_for_all_resumes(resumes, job_desc_embedding, chunk_size=512, overlap_size=10):
    similarity_scores = []
    progress_bar = st.progress(0)
    for idx, resume in enumerate(resumes):
        cleaned_resume = preprocess_text(resume)
        resume_chunks = split_text_into_chunks_with_overlap(cleaned_resume, chunk_size, overlap_size)
        resume_chunk_embeddings = model.encode(resume_chunks)
        final_resume_embedding = np.mean(resume_chunk_embeddings, axis=0)
        similarity_score = cosine_similarity(
            np.array(job_desc_embedding).reshape(1, -1),
            np.array(final_resume_embedding).reshape(1, -1)
        )
        similarity_scores.append(similarity_score[0][0])
        progress_bar.progress((idx + 1) / len(resumes))
    return similarity_scores


def evaluate_resume_ranking(sorted_resumes_df, true_label_column="chosen", top_n=None):
    if true_label_column not in sorted_resumes_df.columns:
        st.error(f"The column '{true_label_column}' does not exist in the sorted resumes.")
        return {}

    true_chosen_indices = sorted_resumes_df[sorted_resumes_df[true_label_column] == 1].index.tolist()
    top_n = max(top_n or len(true_chosen_indices), len(true_chosen_indices))
    predicted_top_indices = sorted_resumes_df.head(top_n).index.tolist()

    TP = set(predicted_top_indices).intersection(true_chosen_indices)
    FP = set(predicted_top_indices).difference(true_chosen_indices)
    FN = set(true_chosen_indices).difference(predicted_top_indices)
    TN = set(sorted_resumes_df.index).difference(TP.union(FP).union(FN))

    metrics = {
        "precision": len(TP) / len(predicted_top_indices) if predicted_top_indices else 0,
        "recall": len(TP) / len(true_chosen_indices) if true_chosen_indices else 0,
        "TP": len(TP),
        "FP": len(FP),
        "FN": len(FN),
        "TN": len(TN)
    }

    metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (
                metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0

    return metrics


# Sidebar navigation
with st.sidebar:
    st.title("Navigation")

    if st.button("📤 Upload & Process", key="nav_upload"):
        st.session_state.current_view = "upload"
    if st.button("📊 Results & Analysis", key="nav_results"):
        st.session_state.current_view = "results"
    if st.button("📈 Evaluation", key="nav_evaluation"):
        st.session_state.current_view = "evaluation"

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.markdown("""
        This AI-powered CV screening tool helps you:
        - Process multiple resumes at once
        - Match candidates to job descriptions
        - Evaluate screening accuracy
        """)

# Main content
if st.session_state.current_view == "upload":
    st.title("📤 Upload & Process CVs")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="upload-text">1. Upload Resumes</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drop PDF or CSV files here",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            help="You can upload multiple PDF resumes or a CSV file containing resume text"
        )

    with col2:
        st.markdown('<p class="upload-text">2. Enter Job Description</p>', unsafe_allow_html=True)
        uploaded_job_desc = st.text_area(
            "Paste the job description here",
            height=200,
            help="Enter the job description to match against the resumes"
        )

    if uploaded_files:
        converter = DocumentConverter()
        extracted_texts = []
        csv_data = None

        with st.spinner("Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file_path = temp_file.name
                    result = converter.convert(temp_file_path)
                    extracted_texts.append(result.document.export_to_markdown())
                elif uploaded_file.type == "text/csv":
                    csv_data = pd.read_csv(uploaded_file)
                    st.session_state.csv_columns = csv_data.columns.tolist()
                    selected_column = st.selectbox(
                        "Select the column containing resumes:",
                        st.session_state.csv_columns
                    )
                    if selected_column:
                        extracted_texts.extend(csv_data[selected_column].dropna().tolist())
                        st.session_state.resumes_data = csv_data

        if extracted_texts:
            st.session_state.resumes_data = pd.DataFrame({"resume_text": extracted_texts})
            if csv_data is not None:
                for col in csv_data.columns:
                    if col not in st.session_state.resumes_data.columns:
                        st.session_state.resumes_data[col] = csv_data[col]

    if st.button("🚀 Process Resumes", help="Click to start processing the uploaded resumes"):
        if not st.session_state.resumes_data.empty and uploaded_job_desc:
            with st.spinner("Processing resumes... This might take a while."):
                resumes = st.session_state.resumes_data["resume_text"].tolist()
                cleaned_job_desc = preprocess_text(uploaded_job_desc)

                job_desc_chunks = split_text_into_chunks_with_overlap(cleaned_job_desc)
                job_desc_chunk_embeddings = model.encode(job_desc_chunks)
                job_desc_embedding = np.mean(job_desc_chunk_embeddings, axis=0)

                similarity_scores = calculate_similarity_for_all_resumes(resumes, job_desc_embedding)

                st.session_state.resumes_data["similarity_score"] = similarity_scores
                st.session_state.sorted_resumes = st.session_state.resumes_data.sort_values(
                    by="similarity_score", ascending=False
                )
                st.session_state.processed = True

                st.success("✅ Processing complete! Go to Results & Analysis to view the rankings.")
        else:
            st.error("Please upload resumes and paste the job description.")

elif st.session_state.current_view == "results" and st.session_state.processed:
    st.title("📊 Results & Analysis")

    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Resumes", len(st.session_state.sorted_resumes))
    with col2:
        st.metric("Average Match Score", f"{st.session_state.sorted_resumes['similarity_score'].mean():.2%}")
    with col3:
        st.metric("Top Match Score", f"{st.session_state.sorted_resumes['similarity_score'].max():.2%}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(st.session_state.sorted_resumes))),  # Convert range to list
        y=st.session_state.sorted_resumes['similarity_score'],
        name='Similarity Score',
        marker_color='rgb(55, 83, 109)'
    ))
    fig.update_layout(
        title='Resume Similarity Scores Distribution',
        xaxis_title='Resume Rank',
        yaxis_title='Similarity Score',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display ranked resumes
    st.subheader("📑 Ranked Resumes")
    for idx, row in st.session_state.sorted_resumes.iterrows():
        with st.expander(f"Rank {idx + 1} - Match Score: {row['similarity_score']:.2%}"):
            st.text_area("Resume Text", row['resume_text'], height=200, key=f"resume_{idx}")

elif st.session_state.current_view == "evaluation" and st.session_state.processed:
    st.title("📈 Evaluation Metrics")

    if st.session_state.csv_columns:
        true_label_column = st.selectbox(
            "Select the column containing true labels:",
            st.session_state.csv_columns
        )

        if st.button("Calculate Metrics"):
            if true_label_column:
                with st.spinner("Calculating evaluation metrics..."):
                    metrics = evaluate_resume_ranking(st.session_state.sorted_resumes, true_label_column)

                    # Display metrics in cards
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>Precision</h3>
                                <h2>{metrics['precision']:.2%}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>Recall</h3>
                                <h2>{metrics['recall']:.2%}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>F1 Score</h3>
                                <h2>{metrics['f1_score']:.2%}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Confusion matrix visualization
                    confusion_matrix = go.Figure(data=go.Heatmap(
                        z=[[metrics['TP'], metrics['FP']],
                           [metrics['FN'], metrics['TN']]],
                        x=['Predicted Positive', 'Predicted Negative'],
                        y=['Actual Positive', 'Actual Negative'],
                        colorscale='Blues'
                    ))
                    confusion_matrix.update_layout(title='Confusion Matrix')
                    st.plotly_chart(confusion_matrix, use_container_width=True)
            else:
                st.error("Please select a column for true labels.")
    else:
        st.info("Upload a CSV file with true labels to enable evaluation.")

else:
    st.info("Please upload and process resumes first.")