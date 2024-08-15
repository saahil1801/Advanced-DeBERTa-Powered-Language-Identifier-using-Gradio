import gradio as gr
import torch
from torch import nn
from torch.optim import AdamW
from transformers import DebertaTokenizer, DebertaModel, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CustomDebertaModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomDebertaModel, self).__init__()
        self.deberta = DebertaModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.deberta.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]  # Take the [CLS] token's hidden state
        x = self.drop(pooled_output)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.drop(x)
        logits = self.fc2(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        return logits

# Load the tokenizer and the model
tokenizer = DebertaTokenizer.from_pretrained('./custom_deberta_language_tokenizer')
model = CustomDebertaModel('microsoft/deberta-base', num_labels=22)
model.load_state_dict(torch.load('./custom_deberta_language_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

description=""" Enter a sentence and the model will predict the language.

This model can identify the following languages:
- Chinese - Thai - English - Japanese - Turkish - Romanian - Urdu - Persian - Korean - Estonian - Russian - Arabic - Portuguese 
- Spanish - Dutch - Pushto - Swedish - Hindi - French - Tamil - Indonesian - Latin
"""

def identify_language(text):
    # Preprocess the text
    encoded_inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # Get predictions
    with torch.no_grad():
        logits = model(
            input_ids=encoded_inputs['input_ids'],
            attention_mask=encoded_inputs['attention_mask']
        )

        predictions = torch.argmax(logits, dim=1)

    # Decode the predictions
    predicted_label = label_encoder.inverse_transform(predictions.cpu().numpy())

    return predicted_label[0]

# Create the Gradio interface
interface = gr.Interface(
    fn=identify_language,
    inputs=gr.components.Textbox(lines=2, placeholder="Enter text here..."),
    outputs=gr.components.Textbox(),
    title="Language Identification",
    description=description
)

if __name__ == "__main__":
    interface.launch()
