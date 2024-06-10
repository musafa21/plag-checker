from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import random
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def extract_full_text(file_path):
    document = Document(file_path)
    full_text = []

    for para in document.paragraphs:
        full_text.append(para.text)
    
    text = '\n'.join(full_text)
    return text

def search_google(phrase):
    api_key = 'Use your own API Key'  # Replace with your actual API key
    search_url = f'https://app.scrapingbee.com/api/v1/store/google?api_key={api_key}&q={phrase}&num=5'
    response = requests.get(search_url)
    
    # Check if the response is empty
    if not response.content:
        return []
    
    try:
        results = response.json().get('organic_results', [])
    except requests.JSONDecodeError:
        results = []

    return results[:5]


def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

def calculate_similarity(doc_text, web_texts):
    texts = [doc_text] + web_texts
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0, 1:]  # Similarity of doc_text with each web_text

def detect_ai_generated_text(text):
    # Tokenize the text
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    # Generate text using the model
    generated_text = model.generate(input_ids, max_length=150, num_return_sequences=1)
    
    # Decode and return the generated text
    decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return decoded_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return redirect(url_for('confirm_upload', filename=file.filename))

@app.route('/confirm_upload/<filename>')
def confirm_upload(filename):
    return render_template('confirm_upload.html', filename=filename)

@app.route('/results/<filename>', methods=['GET', 'POST'])
def results(filename):
    if request.method == 'POST':
        # Perform the similarity check and generate results
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        encodings = ['utf-8', 'latin-1', 'utf-16']
        for encoding in encodings:
            try:
                doc_text = extract_full_text(file_path)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("Could not decode file using available encodings.")
        
        search_results = search_google(doc_text)
        web_texts = [scrape_content(result['link']) for result in search_results if 'link' in result]
        similarity_scores = calculate_similarity(doc_text, web_texts)

        # Create a buffer to store PDF
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Write the uploaded document text to PDF
        p.drawString(100, height - 40, f"Results for {filename}")
        y = height - 80
        for para in doc_text.split('\n'):
            p.drawString(100, y, para)
            y -= 20
            if y < 50:
                p.showPage()
                y = height - 40

        # Check if there are any similarities
        if similarity_scores:
            # Write the similarities to PDF
            y -= 40
            p.drawString(100, y, "Similarities:")
            y -= 20
            for i, score in enumerate(similarity_scores):
                url = search_results[i]['link']
                p.drawString(100, y, f"Similarity: {score * 100:.2f}% - {url}")
                y -= 20
                if y < 50:
                    p.showPage()
                    y = height - 40
        else:
            # If no similarities found, display a message
            y -= 40
            p.drawString(100, y, "No similarities found.")

        # Write the AI-generated text detection result
        y -= 40
        ai_generated_text = detect_ai_generated_text(doc_text)
        p.drawString(100, y, "AI-generated text detection:")
        y -= 20
        p.drawString(100, y, ai_generated_text)

        # Write the overall similarity percentage if similarity scores are available
        if similarity_scores:
            y -= 40
            overall_similarity = sum(similarity_scores) / len(similarity_scores) * 100
            p.drawString(100, y, f"Overall Similarity: {overall_similarity:.2f}%")

        # Save the PDF
        p.save()
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name=f'{filename}_results.pdf', mimetype='application/pdf')
    else:
        # Handle GET request to display the confirmation page
        return render_template('confirm_upload.html', filename=filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

