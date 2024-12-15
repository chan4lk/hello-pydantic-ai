import os
from dotenv import load_dotenv
import PyPDF2
from pptx import Presentation
from business_term_sheet import BusinessTermSheet, BusinessTermExtractor, BusinessTermReviewer, BusinessTermAdmin

# Load environment variables
load_dotenv()

class PitchDeckAnalyzer:
    def __init__(self):
        self.extractor = BusinessTermExtractor()
        self.reviewer = BusinessTermReviewer()
        self.admin = BusinessTermAdmin()
        self.max_retries = 3

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
        return text

    def extract_text_from_pptx(self, pptx_path: str) -> str:
        """Extract text content from a PowerPoint file."""
        text = ""
        try:
            prs = Presentation(pptx_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
        except Exception as e:
            print(f"Error extracting text from PPTX: {e}")
        return text

    def analyze_pitch_deck(self, file_path: str) -> BusinessTermSheet:
        """Analyze pitch deck and extract business terms with review and approval process."""
        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.pptx', '.ppt')):
            text = self.extract_text_from_pptx(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a PDF or PPT/PPTX file.")

        # Initial extraction
        terms = self.extractor.extract(text)
        print("\nInitial extraction completed.")
        
        # Review and retry process
        retry_count = 0
        while retry_count < self.max_retries:
            # Get admin approval
            approved, feedback = self.admin.approve_terms(terms)
            print(f"\nAdmin review - {feedback}")
            
            if approved:
                break
                
            # If not approved and we haven't exceeded retries, review and try again
            print(f"\nAttempting review iteration {retry_count + 1}/{self.max_retries}")
            terms = self.reviewer.review(terms, text)
            retry_count += 1

        return terms

def main():
    analyzer = PitchDeckAnalyzer()
    
    # Example usage
    file_path = "/Users/chandima/repos/resume-analysis/term-sheets/data/pitch/BISTEC Care Deck Short version.pdf"
    try:
        results = analyzer.analyze_pitch_deck(file_path)
        print("\nFinal Extracted Business Terms:")
        print(results.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error analyzing pitch deck: {e}")

if __name__ == "__main__":
    main()
