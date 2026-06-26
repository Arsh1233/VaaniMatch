import json
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Mock WebRTC / GC Speech-to-Text Setup ---
class MockSpeechToText:
    """Mocks Google Cloud Speech-to-Text API with Indic language support."""
    def __init__(self, language_code="hi-IN"):
        self.language_code = language_code
        
    def transcribe(self, audio_chunk):
        # Mocking a transcription of a Hindi audio chunk
        print(f"[SpeechToText] Transcribing audio with language={self.language_code}...")
        return "मेरा नाम राहुल है। मुझे Python और SQL आता है। मैं बैंगलोर में रहता हूँ।"

# --- Pydantic Schema for Slot Filling ---
class ProfileSlots(BaseModel):
    name: Optional[str] = Field(description="The candidate's full name.")
    skills: Optional[list[str]] = Field(description="List of technical or soft skills mentioned.")
    experience: Optional[str] = Field(description="Years of experience or current role.")
    location: Optional[str] = Field(description="City or region where the candidate lives.")

# --- LangChain Service ---
class VoiceToProfileMapper:
    def __init__(self):
        # Assuming OpenAI is configured via environment variables
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=ProfileSlots)
        
        # Prompt to extract information into the JSON schema
        self.extraction_prompt = PromptTemplate(
            template="""
            You are a helpful HR assistant. Extract the following candidate profile information from the transcript.
            If a piece of information is not present in the transcript, leave it as null.
            The transcript may be in Hindi, Tamil, or English, but you must extract standard English representations.
            
            Transcript: {transcript}
            
            {format_instructions}
            """,
            input_variables=["transcript"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        
        self.chain = self.extraction_prompt | self.llm | self.parser

    def process_transcript(self, transcript: str):
        print(f"Processing transcript: '{transcript}'")
        extracted_data = self.chain.invoke({"transcript": transcript})
        
        # Check for missing slots and generate follow-up question
        missing_slots = [k for k, v in extracted_data.items() if not v]
        
        follow_up_question = None
        if missing_slots:
            # Generate a dynamic follow up question
            follow_up_prompt = PromptTemplate.from_template(
                "The following information is missing from the candidate's profile: {missing_slots}. "
                "Write a polite, one-sentence question in Hindi asking for this information."
            )
            follow_up_chain = follow_up_prompt | self.llm
            follow_up_question = follow_up_chain.invoke({"missing_slots": ", ".join(missing_slots)}).content
            
        return {
            "structured_profile": extracted_data,
            "missing_slots": missing_slots,
            "follow_up_action": follow_up_question
        }

# --- Service Execution ---
if __name__ == "__main__":
    print("Initializing Voice-to-Profile Mapper (Dry Run)...")
    
    stt = MockSpeechToText(language_code="hi-IN")
    mapper = VoiceToProfileMapper()
    
    # Simulate receiving an audio chunk via WebRTC
    mock_audio_blob = b"mock_audio_data"
    transcript = stt.transcribe(mock_audio_blob)
    
    # Process transcript
    # NOTE: Since we don't have an active OpenAI API key in this mocked environment,
    # calling mapper.process_transcript() would fail. 
    # We will mock the output of the chain for demonstration purposes.
    
    print("\n--- Expected Output Flow ---")
    mock_extracted_json = {
        "name": "Rahul",
        "skills": ["Python", "SQL"],
        "experience": None,  # Not mentioned in transcript
        "location": "Bangalore"
    }
    
    missing = ["experience"]
    mock_follow_up = "कृपया मुझे अपने काम के अनुभव (experience) के बारे में बताएं।"
    
    result = {
        "structured_profile": mock_extracted_json,
        "missing_slots": missing,
        "follow_up_action": mock_follow_up
    }
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
