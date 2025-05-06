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
from .ollama import is_query_related_to_last

def handle_pdf_upload(request):
    if request.method == "POST" and request.FILES.get("pdf_file"):
        uploaded_file = request.FILES["pdf_file"]
        query = request.POST.get("query", "").strip()
        file_ext = uploaded_file.name.split(".")[-1].lower()

        uploaded_instance = UploadedPDF.objects.create(file=uploaded_file)
        file_path = uploaded_instance.file.path

        # Generate default query if none provided
        if not query:
            if file_ext == "pdf":
                query = f"Summarize the key points or contents of this PDF file: {uploaded_file.name}"
            elif file_ext in ["jpg", "jpeg", "png", "webp"]:
                query = "Describe this image"
            else:
                query = "Analyze this uploaded file"

        if file_ext == "pdf":
            # Always overwrite DB and use current PDF as context
            process_uploaded_pdf(file_path)
            db_path = process_uploaded_pdf(file_path)  # ⬅️ Now gets unique path

            # Use new PDF context always
            knowledge_context = get_relevant_context_from_db(query, db_path)
            full_prompt = generate_rag_prompt(query, "", knowledge_context)

            ai_response = generate_answer_with_ollama(full_prompt)
            ai_response = format_paragraph(ai_response)

            QueryHistory.objects.create(
                query=query,
                response=ai_response,
                pdf_file_name=uploaded_file.name
            )

            return JsonResponse({
                "message": "✅ PDF uploaded and analyzed successfully.",
                "filename": uploaded_file.name,
                "response": ai_response,
            })

        elif file_ext in ["jpg", "jpeg", "png", "webp"]:
            extracted_text = analyze_image_with_llava(file_path, prompt=query)

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

import uuid

def process_uploaded_pdf(pdf_path):
    loaders = [PyPDFLoader(pdf_path)]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)

    embeddings_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'mps'}
    )

    # 🔑 Create unique folder name for each PDF
    unique_id = str(uuid.uuid4())
    persist_dir = f"./chroma_dbs/db_{unique_id}"

    # 🧠 Create vector store for current PDF
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings_function,
        persist_directory=persist_dir
    )
    vectorstore.persist()

    print(f"✅ Vector DB created at: {persist_dir}")
    return persist_dir  # Return so we can use it later

def rag_query(request):
    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        if not query:
            return JsonResponse({"error": "Empty query."}, status=400)

        # 🧠 1. Include recent chat context
        recent_chats = QueryHistory.objects.order_by('-created_at')[:10].values_list("query", "response")
        chat_context = "\n".join([
            f"User: {q}\nAI: {r}" for q, r in reversed(recent_chats) if q and r
        ])

        # 📘 2. Try to retrieve the most recent PDF-based knowledge
        pdf_knowledge = ""
        last_pdf_entry = QueryHistory.objects.filter(pdf_file_name__isnull=False).order_by('-created_at').first()
        if last_pdf_entry:
            pdf_file = UploadedPDF.objects.filter(file__icontains=last_pdf_entry.pdf_file_name).first()
            if pdf_file:
                try:
                    pdf_file_path = pdf_file.file.path
                    db_path = process_uploaded_pdf(pdf_file_path)  # Recreate the vector DB for now
                    pdf_knowledge = get_relevant_context_from_db(query, db_path)
                except Exception as e:
                    print(f"⚠️ Failed to rebuild vector DB: {e}")
                    pdf_knowledge = ""

        # 🧩 3. Build final prompt with context
        full_prompt = generate_rag_prompt(query, chat_context, pdf_knowledge)

        # 🤖 4. Generate the answer
        raw_answer = generate_answer_with_ollama(full_prompt)
        formatted_answer = format_paragraph(raw_answer)

        # 💾 5. Save the conversation
        QueryHistory.objects.create(query=query, response=formatted_answer)

        # 📜 6. Return updated chat history
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


