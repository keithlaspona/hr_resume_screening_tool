import os
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for
from PyPDF2 import PdfReader
import openai

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the OpenAI API key from an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# Function to summarize text using OpenAI API
def summarize_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarize the following text: {text[:1500]}"}],
            max_tokens=300,
            temperature=0.7,
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        return f"Error in summarization: {e}"

# Function to store data in SQLite3
def store_data_in_db(job_description, resumes, scores, summaries):
    try:
        conn = sqlite3.connect('resumes.db')
        c = conn.cursor()
        c.execute(''' 
        CREATE TABLE IF NOT EXISTS resume_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            job_description TEXT, 
            resume_name TEXT, 
            resume_text TEXT, 
            score REAL, 
            summary TEXT
        ) ''')

        for i, resume_text in enumerate(resumes):
            c.execute('''
            INSERT INTO resume_data (job_description, resume_name, resume_text, score, summary)
            VALUES (?, ?, ?, ?, ?) 
            ''', (job_description, f"resume_{i + 1}", resume_text, scores[i], summaries[i]))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_description = request.form["job_description"]
        uploaded_files = request.files.getlist("resumes")

        if uploaded_files and job_description:
            resumes = []
            summaries = []

            for file in uploaded_files:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)  # Ensure file is saved in binary mode
                print(f"Saving file to {filepath}")
                resumes.append(extract_text_from_pdf(filepath))
                summaries.append(summarize_text(resumes[-1]))

            scores = rank_resumes(job_description, resumes)
            store_data_in_db(job_description, resumes, scores, summaries)

            # Save filenames in session
            if 'resumes' not in session:
                session['resumes'] = []
            for file in uploaded_files:
                session['resumes'].append(file.filename)
            session.modified = True

            # Round scores to 2 decimals
            scores = [round(score, 2) for score in scores]

            results = list(zip([file.filename for file in uploaded_files], scores, summaries))
            results.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending

            top_match = results[0] if results else None
            return render_template("index.html", results=results, top_match=top_match)

    return render_template("index.html", results=None, top_match=None)

if __name__ == "__main__":
    app.run(debug=True)