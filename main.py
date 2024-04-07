import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS  # for vector embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF
import json
import requests
from pydub import AudioSegment
import base64


def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text)
    pdf.output("summary.pdf")


load_dotenv()

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # we have read the pdf
        for page in pdf_reader.pages:  # the read pdf will be in form of list and we will now read text form each page
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    # As a professional summary generator, your task is to produce a comprehensive summary on the given 
    # context in approximately 10 pages, with 200 words per page. Ensure that the summary is detailed 
    # and covers all key points.

    As a professional summary generator, your task is to produce a comprehensive summary of each 
    chapter on separate pages, with each page containing approximately 200 words.




    context:\n {context}?\n
    question: \n{question}\n

    answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.33)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "qeustion"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    ## stuff will help in internal text summarization

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )

    print(response)

    st.write("Reply: ", response["output_text"])
    return response["output_text"]


def audio(text):
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZWRhYWU1N2MtNjU1My00Zjc4LWJjN2MtMWU3OTM5YmEyOWY5IiwidHlwZSI6ImFwaV90b2tlbiJ9.gg_dX2JoAs2tfivul7jIKr3lYOCswiftT8cChAd6u8I"
    }

    url = "https://api.edenai.run/v2/audio/text_to_speech"
    payload = {
        "providers": "google,amazon",
        "language": "en-US",
        "option": "FEMALE",
        "text": text,
        "fallback_providers": "",
    }

    response = requests.post(url, json=payload, headers=headers)

    # Check for successful response
    if response.status_code == 200:
        result = json.loads(response.text)
        # Access audio download URL (assuming the response provides a URL)
        audio_url = result.get('google', {}).get('audio_resource_url')
        if audio_url:
            try:
                # Download the audio using requests
                audio_response = requests.get(audio_url)
                if audio_response.status_code == 200:
                    # Save the downloaded audio as bytes
                    audio_bytes = audio_response.content
                    return audio_bytes
                else:
                    print("Error:", audio_response.status_code, "downloading audio")
            except requests.exceptions.RequestException as e:
                print("Error downloading audio:", e)
        else:
            print("Error: Audio URL not found in response.")
    else:
        print("Error:", response.status_code)


def main():
    st.set_page_config("Chat PDF")
    st.image("./gemini.jpg", caption='Sunset', use_column_width=True)
    st.header("Chat with PDF using Gemini ")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        text_input = user_input(user_question)
        # Adding buttons for downloading PDF and audio
        generate_pdf(text_input)
        pdf_bytes = open("summary.pdf", "rb").read()

        # Download PDF button
        st.markdown("<h2 style='text-align: center;'>Download Outputs</h2>", unsafe_allow_html=True)
        if st.button("Download PDF"):
            st.download_button(
                label="Click here to download PDF",
                data=pdf_bytes,
                file_name="summary.pdf",
                mime="application/pdf"
            )

        if text_input:
            audio_bytes = audio(text_input)
            if audio_bytes:
                st.success("Audio generated successfully!")
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                href = f'<a href="data:audio/mp3;base64,{audio_b64}" download="output.mp3">Download Audio File</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Failed to generate audio!")
        else:
            st.warning("Please enter some text.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
