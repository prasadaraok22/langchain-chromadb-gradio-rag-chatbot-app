# ui.py
import gradio as gr
from qabot_util import upload_file_function, qa_function

def build_rag_application():
    with gr.Blocks() as rag_application:
        gr.Markdown("### Form 1: Document Upload (PDF/JSON/CSV formats only)")
        with gr.Row():
            file_input = gr.File(label="Upload your document")
            upload_output = gr.Textbox(label="Status")
        upload_btn = gr.Button("Process File")

        gr.Markdown("---")
        gr.Markdown("### Form 2: Question & Answer")
        with gr.Row():
            question_input = gr.Textbox(label="Enter your question")
            answer_output = gr.TextArea(label="Answer")
        qa_btn = gr.Button("Get Answer")

        upload_btn.click(fn=upload_file_function, inputs=file_input, outputs=upload_output)
        qa_btn.click(fn=qa_function, inputs=question_input, outputs=answer_output)
    return rag_application

if __name__ == "__main__":
    rag_application = build_rag_application()
    rag_application.launch(server_name="0.0.0.0", server_port=7860)