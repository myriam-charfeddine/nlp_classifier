import gradio as gr
from theme_classifier import theme_classifier
import pandas as pd


# def get_themes2(themes_list, subtitles_path, save_path):
#   themes_list_ = themes_list.split(',')
#   theme_classifier_ = theme_classifier(themes_list_)
#   output_df = theme_classifier_.get_all_themes_of_dataset(subtitles_path, save_path)

#   #remove no needed columns
#   theme_list = [theme for theme in themes_list if theme not in 'love']
#   output_df = output_df[theme_list]

#   output_df = output_df[theme_list].sum().reset_index()
#   output_df.columns = ['Theme', 'Score']

#   #make our visualization
#   output_plot = gr.BarPlot(
#     output_df,
#     x='Theme',
#     y='Score',
#     title='The Serie themes',
#     tooltip=['Theme', 'Score'],
#     vertical=False,
#     width=500,
#     height=270
#   )

#   return output_plot
#   # return 'DONE'

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




# def main1():
#     with gr.Blocks() as iface:
#         # Theme Classification Section
#         with gr.Row():
#             with gr.Column():
#                 gr.HTML("<h1>Theme Classification (Zero Shot Claasifiers)</h1>")
#                 with gr.Row():
#                     with gr.Column():
#                         plot = gr.BarPlot()
#                     with gr.Column():
#                         theme_list = gr.Textbox(label="Themes")
#                         subtitles_path = gr.Textbox(label="Subtitles or script Path")
#                         save_path = gr.Textbox(label="Save Path")
#                         get_themes_button =gr.Button("Get Themes")

#                         # output = gr.Textbox(label="Output Box")

                        
#                         get_themes_button.click(get_themes, inputs=[theme_list], outputs=[plot])
#                         # output_df = get_themes('love,betrayal,friendship,dialogue')
#                         # gr.BarPlot(
#                         #     output_df,
#                         #     x="Theme",
#                         #     y="Score",
#                         #     title="Series Themes",
#                         #     tooltip=["Theme","Score"],
#                         #     vertical=False,
#                         #     width=500,
#                         #     height=260
#                         # )

# def main3():
#     with gr.Blocks() as iface:
#         # Theme Classification Section
#         with gr.Row():
#             with gr.Column():
#                 gr.HTML("<h1>Theme Classification (Zero Shot Claasifiers)</h1>")
#                 with gr.Row():
#                     with gr.Column():
#                         plot = gr.BarPlot()
#                     output = gr.Textbox(label="Output Box")
#                     with gr.Column():
#                         theme_list = gr.Textbox(label="Themes")
#                         subtitles_path = gr.Textbox(label="Subtitles or script Path")
#                         save_path = gr.Textbox(label="Save Path")
#                         get_themes_button =gr.Button("Get Themes")
#                         get_themes_button.click(get_themes, inputs=[theme_list,subtitles_path,save_path], outputs=[plot])
#     iface.launch(share=True)

# def main():
#     with gr.Blocks() as iface:
#         # Theme Classification Section
#         with gr.Row():
#             with gr.Column():
#                 gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
                
#                 with gr.Row():
#                     with gr.Column():
#                         # Create an empty plot container, plot will be updated later
#                         # plot = gr.LinePlot()  # Initialize without data
#                         data_display = gr.DataFrame()
                        
#                     # output = gr.Textbox(label="Output Box")
                    
#                     with gr.Column():
#                         theme_list = gr.Textbox(label="Themes")
#                         subtitles_path = gr.Textbox(label="Subtitles or Script Path")
#                         save_path = gr.Textbox(label="Save Path")
#                         get_themes_button = gr.Button("Get Themes")
                        
#                         # Make sure this button doesn't block the UI
#                         get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[data_display])

#     iface.launch(share=True)

def main():
  with gr.Blocks() as iface:
    #Section for Theme Classification 
    with gr.Row():
      with gr.Column():
        gr.HTML('<h1>Theme Classification : Zero Shot Classifiers</h1>')
        with gr.Row():
          with gr.Column():
          #   plot = gr.BarPlot()
            df_output = gr.DataFrame()
          with gr.Column():
            themes_list = gr.Textbox(label = "Themes")
            subtitles_path = gr.Textbox(label = "Subtitles or script Path")
            save_path = gr.Textbox(label = "Save Path")
            get_themes_button = gr.Button("Get Themes")
            # get_themes_button.click(get_themes, inputs=[themes_list, subtitles_path, save_path], outputs=[plot])   
            get_themes_button.click(get_themes, inputs=[themes_list, subtitles_path, save_path], outputs=[df_output]) 
          
            
  iface.launch(share=True)

if __name__ == '__main__':
  main()