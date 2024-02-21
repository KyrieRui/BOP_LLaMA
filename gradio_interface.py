import gradio as gr
from images import logo_svg
from gradio_pdf import PDF


# Gradio application setup
def create_blocks():
    with gr.Blocks(
        css=".logo { "
        "display:flex;"
        "background-color: rgba(3, 91, 143, 0.8);"
        "height: 80px;"
        "border-radius: 8px;"
        "align-content: center;"
        "justify-content: center;"
        "align-items: center;"
        "}"
        ".logo img { height: 305% }"
        ".contain { display: flex !important; flex-direction: column !important; }"
        "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
        "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
        "#col { height: calc(100vh - 112px - 16px) !important; }" 
    ) as blocks:
        with gr.Row():
                    gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div")
       
        with gr.Column():
            
            

            with gr.Row():            
            
                chat_history = gr.Chatbot(
                    label="BOP GPT",
                    value=[], 
                    height=480
                )

                pdf = PDF(label="Upload a PDF", interactive=True)
                
                pdf.upload(lambda f: f, pdf)

        with gr.Row():
            with gr.Column(scale=6):
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Type here to ask your PDF",
                container=False)

            with gr.Column(scale=6):
                submit_button = gr.Button('Send')

            
            
        return blocks, chat_history, text_input, submit_button, 

if __name__ == '__main__':
    demo, chatbot, text_input, submit_button = create_blocks()
    demo.queue()
    demo.launch()