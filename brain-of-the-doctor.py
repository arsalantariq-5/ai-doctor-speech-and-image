# Setup groq api key
import os
import base64
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Convert image to required format

image_path = "acne.jpg"
image_file = open(image_path, "rb")

encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# setup multimodel llm

client = Groq()
query="Is there something wrong with my face?"
model="llama-3.2-90b-vision-preview"

messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": query
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                },
            },
        ],
    }]

chat_completions = client.chat.completions.create(
    messages=messages,
    model=model
)

print(chat_completions.choices[0].message.content)