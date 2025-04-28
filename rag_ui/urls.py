from django.urls import path
from .views import rag_query, handle_pdf_upload, get_current_knowledge, get_chat_history

urlpatterns = [
    path("", rag_query, name="rag_query"),
    path("upload/", handle_pdf_upload, name="upload_pdf"),
    path("current_knowledge/", get_current_knowledge, name="current_knowledge"),
    path("get_chat_history/", get_chat_history, name="get_chat_history"),  # New chat history endpoint
]
