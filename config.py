import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

embedding_model = "models/gemini-embedding-001"
llm_model_name  = "gemini-2.0-flash"
