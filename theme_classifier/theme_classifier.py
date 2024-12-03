import torch
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import os

import sys
import pathlib 
folder_path = pathlib.Path(__file__).parent.resolve() #add the current folder to system to make it accessible
sys.path.append(os.path.join(folder_path,'../')) #out of current folder so we can access other folders like utils and use its modules
from utils import load_subtitles


nltk.download('punkt')
nltk.download('punkt_tab')


class theme_classifier():
    def __init__(self, theme_list):
        self.model_name = 'facebook/bart-large-mnli'
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)
       
    def load_model(self, device):
        theme_classifier = pipeline(
                'zero-shot-classification',
                model = self.model_name,
                device= device
            )

        return theme_classifier
    
    def get_themes_inference(self, script):
        #script tokenization
        script_sentences = sent_tokenize(script)
        
        #script sentences batching
        sentences_batch_size = 20
        scripts_batches = []
        for i in range(0, len(script_sentences), sentences_batch_size):
            sent = ' '.join(script_sentences[i : i+sentences_batch_size])
            scripts_batches.append(sent)

        #script sentences theme classification
        theme_classification_output = self.theme_classifier(
            scripts_batches,
            self.theme_list,
            multi_label=True
        )
        
        #wrangling output : clean, transform data into a structured format
        themes = {}
        for output in theme_classification_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label]=[]
                themes[label].append(score)
        
        themes = {key: np.mean(values) for key, values in themes.items()}
        
        return themes
    
    def get_all_themes_of_dataset(self, dataset_path, save_path=None):
        #read themes inference results dataset if save_path exists
        if save_path is not None and os.path.exists(save_path) :
            df = pd.read_csv(save_path)
            return df
       
        df = load_subtitles(dataset_path)
        # df = df.head(2)
        

        #get theme inference of each row of the subtitles dataset
        themes_classification_output = df['Script'].apply(self.get_themes_inference)

        #convert result to a DataFrame
        df_themes = pd.DataFrame(themes_classification_output.tolist())

        #add the inference results dataset to the original dataset of subtitles
        df[df_themes.columns] = df_themes

        #save result if save_path is specified
        if save_path is not None:
            df.to_csv(save_path, index=False)

        return df



  