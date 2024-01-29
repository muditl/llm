from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from transformers import AutoTokenizer
import torch
import streamlit as st

# reads pdf file
def pdf_read(pdf):

    pdf_reader = PdfReader(pdf)
    pdf_text = ""

    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
        
    return pdf_text


# queries the model to return a summary of the input text
def summarise(chunks, model):
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    summarizer = pipeline("summarization", model=model , tokenizer=tokenizer,  torch_dtype=torch.bfloat16)
    
    summary = summarizer(chunks, max_length=150, min_length=50, do_sample=False)
    
    return summary


# makes a streamlit UI and displays summary of user-uploaded files. 
def main():
    st.set_page_config(page_title="Summarise", page_icon=":book:")
    st.header("Summarizer")
    st.write("PDF/Image")
    
    input = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg","pdf"],accept_multiple_files=False)
    st.header("Summary")
    
    text = ""
    if input is not None:
    
        text = pdf_read(input)
        
        text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 400,
        length_function = len)
        chunks = text_splitter.split_text(text)
        print(f"Split {len(text)} text into {len(chunks)} chunks")
        
        model="facebook/bart-large-cnn"
        # model = "mistralai/Mixtral-8x7B-v0.1"
        
        Summary = summarise(chunks, model)
        
        st.write(Summary)
    
    
if __name__ == '__main__':
    main()