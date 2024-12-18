import torch
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          pipeline)
import pandas as pd
from .cleaner import Cleaner
from sklearn import preprocessing
from datasets import Dataset
from sklearn.model_selection import train_test_split
import gc 
from .training_utils import get_class_weights, compute_metrics
from .custom_trainer import CustomTrainer


class JujetsuClassifier:
    def __init__(self, 
                 model_path,
                 data_path=None,
                 text_column='text',
                 label_column='jutsus',
                 model_name='distilbert/distilbert-base-uncased',
                 test_size=0.2,
                 num_labels=3,
                 huggingface_token=None):
        
        self.model_path=model_path
        self.data_path=data_path
        self.text_column=text_column
        self.label_column=label_column
        self.model_name=model_name
        self.test_size=test_size
        self.num_labels=num_labels
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        self.huggingface_token=huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

        self.tokenizer = self.load_tokenizer()

        if not huggingface_hub.repo_exists(self.model_path):
            if data_path is None:
                raise ValueError('Data path is required to train model since model path does not exist in Huggingface Hub!')
            
            train_data, test_data = self.load_data(self.data_path)

            #we are converting data HuggingFace Dataset to Pandas again to handle skewed dataset issues by applying penalities
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()

            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data)

            self.train_model(train_data, test_data, class_weights)
        
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path):
        model = pipeline('text-classification', model=model_path, return_all_scores=True)
        return model

    def train_model(self, train_data, test_data, class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, 
                                                                   num_labels=self.num_labels,
                                                                   id2label=self.label_dict)
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(output_dir=self.model_path,
                                          learning_rate=2e-4,
                                          per_device_train_batch_size=8,
                                          per_device_eval_batch_size=8,
                                          num_train_epochs=5,
                                          weight_decay=0.01,
                                          evaluation_strategy="epoch",
                                          logging_strategy="epoch",
                                          push_to_hub=True,
                                        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset = train_data,
            eval_dataset = test_data,
            tokenizer = self.tokenizer,
            data_collator=data_collator,
            compute_metrics= compute_metrics
        )

        trainer.set_device(self.device)
        trainer.set_compute_weights(class_weights)
        trainer.train()

        #flush memory
        del trainer, model
        gc.collect()

        if self.device == 'cuda':
            torch.cuda.empty_cache()
    
    def simplify_justsu_type(self, jutsu):
        if 'Ninjutsu' in jutsu:
            return 'Ninjutsu'
        if 'Taijutsu' in jutsu:
            return 'Taijutsu'
        if 'Genjutsu' in jutsu:
            return 'Genjutsu'
        
    def process_text(self, tokenizer, samples):
        return tokenizer(samples['text_cleaned'], truncation=True)

    def load_data (self, data_path):
        data = pd.read_json(data_path, lines=True)
        data['jutsu_type_simplified'] = data['jutsu_type'].apply(self.simplify_justsu_type)
        data['text'] = data['jutsu_name'] + ". " + data['jutsu_description']
        data['jutsus'] = data['jutsu_type_simplified']
        data = data[['text', 'jutsus']]
        data = data.dropna()

        #clean data
        cleaner = Cleaner()
        data['text_cleaned'] = data['text'].apply(cleaner.clean_text)

        #encode labels
        le = preprocessing.LabelEncoder()
        le.fit(data[self.label_column].tolist())
        label_dict = {index:label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
        self.label_dict = label_dict
        data['label'] = le.transform(data['jutsus'].tolist())

        #split data
        train_df, test_df = train_test_split(data, test_size=self.test_size, stratify=data['label'])

        #Convert pandas dataframe to HuggingFace Dataset
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        #processing
        train_dataset_tokenized = train_dataset.map(lambda examples: self.process_text(self.tokenizer, examples), batched=True)
        test_dataset_tokenized = test_dataset.map(lambda examples: self.process_text(self.tokenizer, examples), batched=True)
        
        return train_dataset_tokenized, test_dataset_tokenized

            
    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return tokenizer
    
    def classify_jutsu(self, text):
        model_output = self.model(text)
        predictions = self.postprocess(model_output)
        return predictions
    
    def postprocess(self, model_output):
        output = []
        for pred in model_output:
            label = max(pred, key=lambda x : x['score'])['label']
            output.append(label)

        return output


            
            