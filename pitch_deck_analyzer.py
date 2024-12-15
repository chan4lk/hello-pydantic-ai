from typing import List, Optional, Dict
import PyPDF2
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os
from business_term_sheet import BusinessTermSheet
from dotenv import load_dotenv
import json
# Load environment variables
load_dotenv()

class PitchDeckAnalyzer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None

    def extract_text_from_pdf(self, file_path: str) -> List[Document]:
        documents = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_num + 1}
                    )
                    documents.append(doc)
        return documents

    def extract_text_from_pptx(self, file_path: str) -> List[Document]:
        documents = []
        presentation = Presentation(file_path)
        for slide_num, slide in enumerate(presentation.slides):
            text_content = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_content.append(shape.text)
            if text_content:
                doc = Document(
                    page_content=" ".join(text_content),
                    metadata={"source": file_path, "slide": slide_num + 1}
                )
                documents.append(doc)
        return documents

    def create_vector_store(self, file_path: str) -> None:
        if file_path.lower().endswith('.pdf'):
            documents = self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.pptx'):
            documents = self.extract_text_from_pptx(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a PDF or PPTX file.")

        split_docs = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)

    def query_field(self, field_name: str, description: str) -> str:
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please call create_vector_store first.")

        # Query the vector store for relevant chunks
        query = f"Find information about {field_name}: {description}"
        docs = self.vector_store.similarity_search(query, k=3)
        
        if not docs:
            return "Not mentioned"

        # Combine the relevant chunks into context
        context = "\n".join([doc.page_content for doc in docs])
        
        # Prepare the prompt for GPT
        prompt = f"""Based on the following context from a pitch deck, extract the {field_name}. 
        If the information is not explicitly mentioned, respond with 'Not mentioned'.
        
        Context:
        {context}
        
        {field_name}:"""

        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise business analyst. Extract specific information from pitch deck content. If information is not explicitly mentioned, respond with 'Not mentioned'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        
        extracted_value = response.choices[0].message.content.strip()
        return "Not mentioned" if not extracted_value else extracted_value

    def analyze_pitch_deck(self, file_path: str) -> BusinessTermSheet:
        # Initialize vector store
        self.create_vector_store(file_path)
        
        # Extract each field using RAG
        extracted_data = {}
        for field_name, field_info in BusinessTermSheet.model_fields.items():
            description = field_info.description or field_name
            extracted_data[field_name] = self.query_field(field_name, description)
        
        # Create and return BusinessTermSheet instance
        return BusinessTermSheet(**extracted_data)

def main():
    analyzer = PitchDeckAnalyzer()
    
    # Example usage
    file_path = "/Users/chandima/repos/resume-analysis/term-sheets/data/pitch/BISTEC Care Deck Short version.pdf"
    try:
        results = analyzer.analyze_pitch_deck(file_path)
        print("\nExtracted Business Terms:")
        print(json.dumps(results.model_dump(), indent=2))
    except Exception as e:
        print(f"Error analyzing pitch deck: {e}")

if __name__ == "__main__":
    main()
