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
from dataclasses import dataclass

# Load environment variables
load_dotenv()

@dataclass
class FieldResult:
    output_value: str
    is_mentioned: bool
    confidence: int

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
        try:
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
                print(f"Extracted {len(documents)} pages from PDF")
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
        return documents

    def extract_text_from_pptx(self, file_path: str) -> List[Document]:
        documents = []
        try:
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
            print(f"Extracted {len(documents)} slides from PPTX")
        except Exception as e:
            print(f"Error extracting text from PPTX: {e}")
        return documents

    def create_vector_store(self, file_path: str) -> None:
        try:
            if file_path.lower().endswith('.pdf'):
                documents = self.extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.pptx'):
                documents = self.extract_text_from_pptx(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide a PDF or PPTX file.")

            if not documents:
                raise ValueError("No text content extracted from the document")

            split_docs = self.text_splitter.split_documents(documents)
            print(f"Created {len(split_docs)} text chunks")
            
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            print("Vector store created successfully")
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise

    def query_field(self, field_name: str, description: str) -> FieldResult:
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized. Please call create_vector_store first.")

            # Query the vector store for relevant chunks
            query = f"Find information about {field_name}: {description}"
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=3)
            
            if not docs_with_scores:
                print(f"No relevant content found for {field_name}")
                return FieldResult(
                    output_value="Not mentioned",
                    is_mentioned=False,
                    confidence=0
                )

            # Calculate average similarity score
            avg_similarity = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
            # Convert similarity score to confidence percentage (0-100)
            confidence = int(min(100, max(0, (1 - avg_similarity) * 100)))

            # Combine the relevant chunks into context
            context = "\n".join([doc.page_content for doc, _ in docs_with_scores])
            
            # Prepare the prompt for GPT
            prompt = f"""Based on the following context from a pitch deck, extract the {field_name}. 
            The field description is: {description}
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
            print(f"Extracted {field_name}: {extracted_value} (Confidence: {confidence}%)")
            
            return FieldResult(
                output_value=extracted_value,
                is_mentioned=extracted_value != "Not mentioned",
                confidence=confidence
            )

        except Exception as e:
            print(f"Error extracting {field_name}: {e}")
            return FieldResult(
                output_value="Error during extraction",
                is_mentioned=False,
                confidence=0
            )

    def analyze_pitch_deck(self, file_path: str) -> Dict[str, Dict[str, any]]:
        try:
            # Initialize vector store
            print(f"\nAnalyzing pitch deck: {file_path}")
            self.create_vector_store(file_path)
            
            # Extract each field using RAG
            extracted_data = {}
            for field_name, field_info in BusinessTermSheet.model_fields.items():
                description = field_info.description or field_name
                print(f"\nExtracting {field_name}...")
                result = self.query_field(field_name, description)
                extracted_data[field_name] = {
                    "output_value": result.output_value,
                    "is_mentioned": result.is_mentioned,
                    "confidence": result.confidence
                }
            
            return extracted_data
            
        except Exception as e:
            print(f"Error analyzing pitch deck: {e}")
            raise

def main():
    analyzer = PitchDeckAnalyzer()
    file_path = "/Users/chandima/repos/resume-analysis/term-sheets/data/pitch/BISTEC Care Deck Short version.pdf"
    
    try:
        results = analyzer.analyze_pitch_deck(file_path)
        print("\nExtracted Business Terms:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error analyzing pitch deck: {e}")

if __name__ == "__main__":
    main()
