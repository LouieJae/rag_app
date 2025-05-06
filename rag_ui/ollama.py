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
        You are a knowledgeable and friendly assistant. You always:
        - Respond clearly and concisely
        - Use **bold** for important terms
        - Use bullet points (•) or numbered lists where appropriate
        - Highlight key facts
        - Maintain a helpful and engaging tone

        {f"Here is the previous conversation:\n{chat_context}" if chat_context else ""}
        {f"\nHere is some related knowledge:\n{knowledge_context}" if knowledge_context else ""}

        Now answer the following user question:
        **{query}**

        Respond below with clarity and structure:
        """
    return prompt


# 🧠 2. Get relevant context from the Chroma database
def get_relevant_context_from_db(query, db_path):
    context = ""
    embeddings_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'mps'}
    )
    vector_db = Chroma(
        persist_directory=db_path,
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
    payload = {"model": "llama3", "prompt": prompt}
    
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
    return response.strip()  # Preserve \n characters for frontend formatting

from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

def is_query_related_to_last(current_query, last_query, threshold=0.65):
    if not current_query or not last_query:
        return False

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'mps'}
    )

    vectors = embeddings.embed_documents([current_query, last_query])
    cosine_sim = sum(a * b for a, b in zip(vectors[0], vectors[1])) / (
        sum(a * a for a in vectors[0])**0.5 * sum(b * b for b in vectors[1])**0.5
    )

    return cosine_sim > threshold