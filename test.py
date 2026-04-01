from dotenv import load_dotenv
import os

# load env file
load_dotenv()

# get api key
api_key = os.getenv("Groq_api_key")

print(api_key)