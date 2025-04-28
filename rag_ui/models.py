from django.db import models

class QueryHistory(models.Model):
    query = models.TextField(blank=True, null=True)
    response = models.TextField(blank=True, null=True)
    image = models.ImageField(upload_to="uploads/", blank=True, null=True)
    pdf_file_name = models.CharField(max_length=255, blank=True, null=True)  # ✅ add this
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        if self.query:
            return self.query[:50]
        elif self.pdf_file_name:
            return f"PDF Upload: {self.pdf_file_name}"
        else:
            return "Image Upload"

class UploadedPDF(models.Model):
    file = models.FileField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name