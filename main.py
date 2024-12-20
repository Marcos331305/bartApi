import fitz
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from fastapi import FastAPI, UploadFile, File
from io import BytesIO

# Check if MPS (Metal Performance Shaders) is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Initialize FastAPI app
app = FastAPI()

# Extract text from the PDF document
def extract_text_from_pdf(pdf_bytes):
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Split text into smaller chunks
def split_text_into_chunks(text, max_length=1024):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunks.append(" ".join(words[i:i + max_length]))
    return chunks

# Summarize the extracted text using BART
def summarize_text(text):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    model = model.to(device)

    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Main API endpoint to summarize PDF
@app.post("/summarize/")
async def summarize_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    extracted_text = extract_text_from_pdf(pdf_bytes)
    chunks = split_text_into_chunks(extracted_text, max_length=512)
    
    summaries = [summarize_text(chunk) for chunk in chunks]
    final_summary = " ".join(summaries)
    return {"summary": final_summary}
