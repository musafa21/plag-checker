
---

# Plagiarism Checker

## Description
This Flask-based web application allows users to upload documents and check them for plagiarism. The application uses various techniques such as web scraping and cosine similarity to compare the uploaded document with content available on the web. Additionally, it leverages a pre-trained GPT-2 language model to detect AI-generated text within the document.

## Features
- Upload documents in various formats (e.g., DOCX).
- Compare the uploaded document with content available on the web.
- Detect AI-generated text within the document using GPT-2.
- Generate a PDF report with plagiarism results in seconds.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/username/plagiarism-checker.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```
   python app.py
   ```

## Usage
1. Access the application through a web browser.
2. Upload the document you want to check for plagiarism.
3. View the plagiarism results, including similarity scores and AI-generated text detection.
4. Download the generated PDF report for further analysis.

## Dependencies
- Flask: Web framework for building the application.
- requests: HTTP library for making requests to web servers.
- BeautifulSoup: Library for web scraping.
- scikit-learn: Library for machine learning algorithms.
- python-docx: Library for reading and writing Microsoft Word files.
- reportlab: Library for PDF generation.
- transformers: Library for natural language processing tasks, including GPT-2.
- torch: PyTorch library for deep learning.


### NB: This is an experimental feature in development. Please be cautious (if you ever try it) And this is is also for education purposes.):-
---