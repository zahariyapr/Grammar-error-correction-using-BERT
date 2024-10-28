from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForMaskedLM
import torch
import os

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Load the saved model state dict from the 'models' folder, mapping to CPU
model_path = os.path.join('models', 'grammar_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def correct_grammar(sentence):
    """Corrects the grammar of a sentence using the loaded model."""
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    corrected_sentence = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return corrected_sentence

@app.route('/', methods=['GET', 'POST'])
def index():
    corrected_sentence = ''
    if request.method == 'POST':
        sentence = request.form['sentence']
        corrected_sentence = correct_grammar(sentence)
    return render_template('index.html', corrected_sentence=corrected_sentence)

if __name__ == '__main__':
    app.run(debug=True)
