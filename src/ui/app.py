import gradio as gr
from src.core.genesis_pipeline import genesis   # import global instance only

def create_video(prompt: str, duration_seconds: float = 5.0):
    return genesis.generate(prompt, int(duration_seconds))

with gr.Blocks(title="Genesis – Dein lokaler Tutoriumsvideo-Generator") as iface:
    gr.Markdown("# Genesis – BigBrother 2.0")
    gr.Markdown("Nur Prompt → perfektes Tutoriumsvideo. Trainierbar auf euren Vorlesungen.")

    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            lines=3,
            value="Weiße Folie, Titel 'Maschinelles Lernen', drei Bulletpoints erscheinen nacheinander mit Fade-In, schwarze Schrift auf weißem Hintergrund, saubere Zoom-Animationen, 8 Sekunden"
        )
    with gr.Row():
        duration = gr.Slider(3, 30, value=8, step=1, label="Dauer in Sekunden")
        submit = gr.Button("Video generieren", variant="primary")

    video_output = gr.Video(label="Dein Tutorium-Video")

    submit.click(fn=create_video, inputs=[prompt, duration], outputs=video_output)
    prompt.submit(fn=create_video, inputs=[prompt, duration], outputs=video_output)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860, share=False)