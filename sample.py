import PyPDF2
import pytesseract
from PIL import Image
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from pathlib import Path
import streamlit as st
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
import torch


def get_file_extension(file_path):
    return Path(file_path).suffix


def pdf_read(pdf):
    pdf_reader = PyPDF2.PdfReader(pdf)
    pdf_text = ""

    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    return pdf_text


def image_text_read(image):
    config_hinglish = r'--oem 3 --psm 6 -l hin+eng'
    image_open = Image.open(image)
    image_text = pytesseract.image_to_string(image_open, config=config_hinglish)
    return image_text


def find_pageNo(output_text, text_in_page_Dictionary):
    for page_num, text in text_in_page_Dictionary.items():
        if output_text in text:
            return page_num + 1


def search_pdfs(output_text, multiple_pdfs):
    final = []
    for PDFS in multiple_pdfs:
        with open(PDFS, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(PDFS)
            text_in_page_dictionary = {}
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                formatted_text = ' '.join(text.split())
                text_in_page_dictionary[page_num] = formatted_text

            page_num = find_pageNo(output_text, text_in_page_dictionary)
            if page_num:
                final.append((PDFS, page_num))

    return final


def make_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        is_separator_regex=False)
    chunks = text_splitter.split_text(text)

    return chunks


def vector_database(chunks):
    embeddings = HuggingFaceBgeEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    return vectorstore


def get_conversation_chain(vectorstore, model, docs, question):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    torch.backends.cuda.max_split_size_mb = 128

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16) # CUDA

    tokenizer = AutoTokenizer.from_pretrained(model, model_kwargs={'temperature': 0}, quantization_config=bnb_config)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16, # CUDA
        device_map="auto"
    )
    torch.manual_seed(4)

    # Build prompt
    template = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, 
    just say that you don't know, don't try to make up an answer.Avoid irrelevant answer.
    Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    I don't want you to make your own code and show that. Only tell the relevant result.
    Always answer in sentences. Don't answer in one word.
    For large text, try to keep the answer in bullets form with key topic been highlighted.
    The answer should be relevant to the context.
    
    Context: {docs}
    Question: {question}
    """

    sequences = pipe(
        template,
        max_new_tokens=1500,  # max number of tokens to generate in the output
        do_sample=True,
        top_p=0.20,  # select from top tokens whose probability add up to 20%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=sequences)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        retriever=vectorstore.as_retriever(),
        memory=memory,
        llm=llm
    )

    for seq in sequences:
        result = f"Result: {seq['generated_text']}"

    return conversation_chain, result


def main():
    st.set_page_config(page_title="Q/A", page_icon=":book:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    st.header("Ask your question")
    st.write("PDF/Image")
    _input = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)
    question = st.text_input("Ask your question")
    button = st.button("Process")

    # variable is never used
    text = ""
    if button:

        all_text = ""
        if _input is not None:

            multiple_pdfs = [File.name for File in _input]

            for files in _input:
                file_extension = get_file_extension(files.name)

                if file_extension == ".pdf":
                    text = pdf_read(files)
                else:
                    text = image_text_read(files)

                all_text += text
                # variable is never used
                formatted_all_text = ' '.join(all_text.split())

            if question:

                model = "psmathur/orca_mini_3b"
                # model = "tiiuae/falcon-7b-instruct"
                # model = "tiiuae/falcon-40b"
                # model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

                chunks = make_chunks(all_text)

                print(f"Split {len(all_text)} text into {len(chunks)} chunks")

                vectorstore = vector_database(chunks)
                print(vectorstore)

                docs = vectorstore.similarity_search_with_score(question, k=12)

                st.session_state.conversation, result = get_conversation_chain(vectorstore, model, docs, question)

                st.write(result)

                output_text = (docs[0])[0].page_content
                formatted_output_text = ' '.join(output_text.split())
                print(formatted_output_text)

                results = search_pdfs(formatted_output_text, multiple_pdfs)

                if results:
                    st.write("Text found in:")
                    for pdf, page_num in results:
                        st.write(f"- {pdf} (page {page_num})")
                else:
                    st.write("Text not found in any PDF.")

            else:
                st.write("No question asked! Please enter a question!")

            print(torch.cuda.memory_summary(device=None, abbreviated=False))


if __name__ == '__main__':
    main()
