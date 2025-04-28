import os
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.conf import settings
from .models import UploadedPDF
from .models import QueryHistory, UploadedPDF
from .ollama import get_relevant_context_from_db, generate_rag_prompt, generate_answer_with_ollama, format_paragraph,analyze_image_with_llava

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
latest_knowledge = "No knowledge uploaded"

def handle_pdf_upload(request):
    if request.method == "POST" and request.FILES.get("pdf_file"):
        uploaded_file = request.FILES["pdf_file"]
        query = request.POST.get("query", "") or "No query provided."
        file_ext = uploaded_file.name.split(".")[-1].lower()

        uploaded_instance = UploadedPDF.objects.create(file=uploaded_file)
        file_path = uploaded_instance.file.path

        # If it's a PDF
        if file_ext == "pdf":
            # Step 1: Insert uploaded PDF into Vectorstore
            process_uploaded_pdf(file_path)

            # Step 2: Fetch knowledge
            knowledge_context = get_relevant_context_from_db(query)

            # Step 3: Build prompt
            full_prompt = generate_rag_prompt(query, "", knowledge_context)

            # Step 4: Generate response
            ai_response = generate_answer_with_ollama(full_prompt)
            ai_response = format_paragraph(ai_response)

            # ✅ Step 5: Save properly INCLUDING pdf_file_name
            QueryHistory.objects.create(
                query=query,
                response=ai_response,
                pdf_file_name=os.path.basename(uploaded_file.name)  # ✅ save the file name here
            )

            return JsonResponse({
                "message": "✅ PDF uploaded and analyzed successfully.",
                "filename": uploaded_file.name,
                "response": ai_response,
            })

        elif file_ext in ["jpg", "jpeg", "png", "webp"]:
            extracted_text = analyze_image_with_llava(file_path, prompt=query or "Describe this image")

            QueryHistory.objects.create(
                query=query,
                response=extracted_text,
                image=uploaded_instance.file
            )

            return JsonResponse({
                "message": "✅ Image uploaded and analyzed successfully.",
                "image_url": uploaded_instance.file.url,
                "extracted_text": extracted_text
            })

        else:
            return JsonResponse({"error": "Unsupported file type."}, status=400)

    return JsonResponse({"error": "No file uploaded."}, status=400)



def get_current_knowledge(request):
    """Fetches the most recent uploaded file and displays only the file name."""
    last_uploaded = UploadedPDF.objects.order_by('-uploaded_at').first()

    if last_uploaded:
        file_name = os.path.basename(last_uploaded.file.name)  # Extract file name only
        current_knowledge = file_name
    else:
        current_knowledge = "No knowledge uploaded"

    return JsonResponse({"current_knowledge": current_knowledge})

def process_uploaded_pdf(pdf_path):
    """Processes uploaded PDF and adds it to the existing Chroma vector database."""
    loaders = [PyPDFLoader(pdf_path)]
    docs = []

    for loader in loaders:
        docs.extend(loader.load())

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)

    # Create embedding function
    embeddings_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'mps'})

    # ⚡ Load existing vectorstore
    vectorstore = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embeddings_function)

    # 📥 Add new documents
    vectorstore.add_documents(docs)

    # 💾 Save updates
    vectorstore.persist()

    print(f"✅ Added {len(docs)} chunks to existing Chroma database.")


def rag_query(request):
    """Handles user queries and fetches chat history along with knowledge source."""
    if request.method == "POST":
        query = request.POST.get("query", "").strip()

        if query:
            # 🧠 Retrieve recent chat history (last 100 interactions for context)
            recent_chats = QueryHistory.objects.order_by('-created_at')[:100].values_list("query", "response")

            # 📚 Format chat history
            chat_context = "\n".join([f"User: {q}\nAI: {r}" for q, r in recent_chats if q and r])

            # 📖 Retrieve relevant knowledge from the vector database
            knowledge_context = get_relevant_context_from_db(query)

            # 🧩 Combine chat history + knowledge
            full_prompt = generate_rag_prompt(query, chat_context, knowledge_context)

            # 🚀 Generate the AI answer using Ollama
            raw_answer = generate_answer_with_ollama(full_prompt)

            # 📝 Format the AI's answer nicely
            formatted_answer = format_paragraph(raw_answer)

            # 🖊️ Save the new query and answer to QueryHistory
            QueryHistory.objects.create(
                query=query,
                response=formatted_answer
                # No image here since this is a text query
            )

            # 🗂 Fetch updated chat history to display
            chat_history = QueryHistory.objects.order_by('-created_at')[:10].values("query", "response", "image", "created_at")

            return JsonResponse({
                "query": query,
                "response": formatted_answer,
                "chat_history": list(chat_history),
            })

    return render(request, "rag_ui/index.html")


def get_chat_history(request):
    chat_history = QueryHistory.objects.order_by('-created_at')[:10].values("query", "response", "image", "pdf_file_name", "created_at")
    return JsonResponse({"chat_history": list(chat_history)})


