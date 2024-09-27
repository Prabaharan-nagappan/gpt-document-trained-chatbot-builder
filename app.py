import os
from flask import Flask, request, jsonify
import pdfplumber
import docx2txt
import markdown2
from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the Hugging Face model (you can choose another model if needed)
qa_model = pipeline("question-answering", model="google/flan-t5-base")

# Route to check if the server is running
@app.route('/')
def index():
    return "Welcome to the GPT-Document-Trained-Chatbot!"

# Route for document upload and text extraction
@app.route('/upload', methods=['POST'])
def upload_document():
    file = request.files['document']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Extract text based on file type
    if file.filename.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            text = ''.join([page.extract_text() for page in pdf.pages])
    elif file.filename.endswith('.docx'):
        text = docx2txt.process(file_path)
    elif file.filename.endswith('.md'):
        with open(file_path, 'r') as f:
            text = markdown2.markdown(f.read())
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    return jsonify({"extracted_text": text})

# Route for asking the chatbot a question based on extracted text
@app.route('/ask', methods=['POST'])
def ask_bot():
    data = request.json
    question = data.get("question")
    context = data.get("context")

    if not question or not context:
        return jsonify({"error": "Question and context are required"}), 400

    # Use the Hugging Face model to answer the question based on the document text
    result = qa_model(question=question, context=context)
    answer = result['answer']

    return jsonify({"response": answer})

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Run the Flask application
    app.run(debug=True)
