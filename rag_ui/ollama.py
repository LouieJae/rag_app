import json
import requests
import base64

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Ollama server endpoints
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# 📚 1. New: generate_rag_prompt that accepts query, chat_history, and knowledge
def generate_rag_prompt(query, chat_context, knowledge_context):
    prompt = f"""
You are a helpful and informative assistant that answers questions using the information provided in the reference context below.
Always respond completely, explain complex concepts in a simple and friendly tone, and provide all relevant background information.
If information from the context is not necessary, you may ignore it.

Here is the previous conversation:
{chat_context}

Here is additional knowledge from documents:
{knowledge_context}

Now, based on everything above, answer the following question:
'{query}'

Answer:
"""
    return prompt

# 🧠 2. Get relevant context from the Chroma database
def get_relevant_context_from_db(query):
    context = ""
    embeddings_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': 'mps'}
    )
    vector_db = Chroma(
        persist_directory="./chroma_db_nccn", 
        embedding_function=embeddings_function
    )
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

# 🖼️ 3. Analyze uploaded images with LLaVA model
def analyze_image_with_llava(image_path, prompt="Extract the text from this image"):
    print("📸 Analyzing image with LLaVA:", image_path)

    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "model": "llava",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_img]
            }
        ]
    }

    response = requests.post(OLLAMA_CHAT_URL, json=payload, stream=True)

    if response.status_code == 200:
        content = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    content += data.get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    continue
        return content.strip() if content else None
    else:
        print("❌ LLaVA failed:", response.status_code, response.text)
        return None

# 🤖 4. Generate an answer using the Ollama model
def generate_answer_with_ollama(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {"model": "qwen2.5-coder:3b", "prompt": prompt}
    
    response = requests.post(OLLAMA_URL, headers=headers, json=payload, stream=True)
    
    if response.status_code == 200:
        full_response = ""

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    full_response += data.get("response", "")
                except json.JSONDecodeError:
                    continue

        return full_response.strip()
    else:
        return f"Error: {response.status_code} {response.text}"

# 🧹 5. Small utility: format paragraph nicely
def format_paragraph(response):
    return response.replace("\n", " ").strip()
