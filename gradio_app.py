import gradio as gr
from theme_classifier import theme_classifier
import pandas as pd
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import JujetsuClassifier
# from chatbot import CharacterChatbot
import os
from dotenv import load_dotenv
load_dotenv()

def get_themes(theme_list_str,subtitles_path,save_path):
    theme_list = theme_list_str.split(',')
    theme_classifier_ = theme_classifier(theme_list)
    output_df = theme_classifier_.get_all_themes_of_dataset(subtitles_path,save_path)

    # Remove dialogue from the theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]

    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme','Score']

    # output_chart = gr.BarPlot(
    #     output_df,
    #     x="Theme",
    #     y="Score",
    #     title="Series Themes",
    #     tooltip=["Theme","Score"],
    #     vertical=False,
    #     width=500,
    #     height=260
    # )
    return output_df.sort_values(by="Score", ascending=False)

def get_character_network(subtitles_path, ner_save_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_all_ners_inference(subtitles_path, ner_save_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.character_network_generator(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html

def classify_text(model_path, data_path, text_to_classify):
    jutsu_classifier = JujetsuClassifier(model_path = model_path, 
                                        data_path = data_path,
                                        huggingface_token=os.getenv('huggingface_token'))
    output = jutsu_classifier.classify_jutsu(text_to_classify)
    # output = output[0]
    return output

def chat_with_character_chatbot(message, history):
    chatbot = CharacterChatbot('',   # Ex (check ur huggingface user name): MyriamCH/Naruto_Llama-3-8B
                               huggingface_token=os.getenv('huggingface_token')) 
    
    output = chatbot.chat(message, history)
    output = output['content'].strip()
    return output


def main():
  with gr.Blocks() as iface:

    #Theme Classification Section
    with gr.Row():
      with gr.Column():
        gr.HTML('<h1>Theme Classification : Zero Shot Classifiers</h1>')
        with gr.Row():
          with gr.Column():
          #   plot = gr.BarPlot()
            df_output = gr.DataFrame()
          with gr.Column():
            themes_list = gr.Textbox(label = "Themes")
            subtitles_path = gr.Textbox(label = "Subtitles")
            save_path = gr.Textbox(label = "Save Path")
            get_themes_button = gr.Button("Get Themes")
            # get_themes_button.click(get_themes, inputs=[themes_list, subtitles_path, save_path], outputs=[plot]) 
            get_themes_button.click(get_themes, inputs=[themes_list, subtitles_path, save_path], outputs=[df_output]) 

    #Character Network Section 
    with gr.Row():
      with gr.Column():
        gr.HTML('<h1>Character Network : NERs and Graph</h1>')
        with gr.Row():
          with gr.Column():
            network_html = gr.HTML()
          with gr.Column():
            subtitles_path = gr.Textbox(label = "Subtitles")
            ner_save_path = gr.Textbox(label = "NERs Save Path")
            network_generator_button = gr.Button("Get Character Network")
            network_generator_button.click(get_character_network, inputs=[subtitles_path, ner_save_path], outputs=[network_html]) 

    #Text Classifiaction using LLMs Section 
    with gr.Row():
      with gr.Column():
        gr.HTML('<h1>Text Classifiaction using LLMs</h1>')
        with gr.Row():
          with gr.Column():
            text_classification_output = gr.Textbox(label='Text Classification Output')
          with gr.Column():
            model_path = gr.Textbox(label='Model Path')
            data_path = gr.Textbox(label='Data Path')
            text_to_classify = gr.Textbox(label='Text input')
            classify_text_button = gr.Button("Clasify Text (Jutsu)")
            classify_text_button.click(classify_text, inputs=[model_path, data_path, text_to_classify], outputs=[text_classification_output])

    # #Character Chatbot Section using Llama3
    # with gr.Row():
    #   with gr.Column():
    #     gr.HTML('<h1>Chat With The Serie\'s Main Character</h1>')
    #     gr.ChatInterface(chat_with_character_chatbot)

          
          
            
  iface.launch(share=True)

if __name__ == '__main__':
  main()