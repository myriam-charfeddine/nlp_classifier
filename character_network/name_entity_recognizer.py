import spacy
from nltk import sent_tokenize
import pandas as pd
from ast import literal_eval
import os 
import sys
import pathlib

folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles

class NamedEntityRecognizer:
   def __init__(self) :
      self.nlp_model = self.load_model()

   def load_model(self):
      nlp_model = spacy.load("en_core_web_trf")
      return nlp_model
   
   def get_ner_inference(self, script):
      sentences = sent_tokenize(script)
      
      ner_output = []
      for sentence in sentences:
         doc = self.nlp_model(sentence)
         ner = set()
         for entity in doc.ents:
               if entity.label_ == 'PERSON':
                  ner.add(entity.text.split(" ")[0].strip())

         ner_output.append(ner)

      return ner_output
   
   def get_all_ners_inference(self, dataset_path, save_path):
      if save_path is not None and os.path.exists(save_path):
         df= pd.read_csv(save_path) #the Ners column contains list as value for each row, which is not supported by a pandas CSV file, so it's gonna be auto converted to a string!!
         df['Ners'] = df['Ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x) #we use literal_eval to convert x back into a list if it's a string, else if x is already a list we leave it unchnageable
         return df
      
      #load subtitles dataset
      df = load_subtitles(dataset_path)
      
      #Run inference to extract entities from loded dataset (df)
      df['Ners'] = df['Script'].apply(self.get_ner_inference)

      if save_path is not None:
         df.to_csv(save_path, index=False)

      return df
   
