from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import openai

app = FastAPI()

# Set up OpenAI API key
openai.api_key = "org-JQ2lt2CyWKNjiYHklaXau7nd"

# Define a route for the chatbot
@app.post("/chat")
async def chat(message: str = Form(...)):
    # Use the OpenAI API to generate a response
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=message,
        max_tokens=2048,
        temperature=0.5,
    )

    # Return the response as HTML
    return HTMLResponse(content=f"<html><body><p>{response['choices'][0]['text']}</p></body></html>")

# Define a route for the chatbot interface
@app.get("/")
def read_root():
    return HTMLResponse(content="""
    <html>
    <body>
    <h1>Chatbot</h1>
    <form action="/chat" method="post">
    <input type="text" name="message" placeholder="Type a message...">
    <input type="submit" value="Send">
    </form>
    </body>
    </html>
    """)

# Fine-tuning the model
def fine_tune_model():
    # Prepare your training data
    training_data = [
        {"prompt": "Translate to French: Hello, how are you?", "completion": "Bonjour, comment ça va?"},
        {"prompt": "Translate to French: What is your name?", "completion": "Quel est ton nom?"}
    ]

    # Fine-tune the model
    response = openai.FineTune.create(training_file="file-xxxxxxxxxxxxxxxxxxxxxx", model="davinci", n_epochs=4)
    print("Fine-tune job ID:", response["id"])

# Creating an image
def create_image():
    response = openai.Image.create(prompt="A futuristic city skyline at sunset", n=1, size="1024x1024")
    image_url = response['data'][0]['url']
    print(image_url)

#Creating an embedding
def create_embedding():
    response = openai.Embedding.create(
        input="This is an example sentence for embedding.",
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    print(embedding)

