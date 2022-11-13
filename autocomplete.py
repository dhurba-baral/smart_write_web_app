from flask import Flask, request,render_template,url_for
from functions.preprocess import sentence_into_words, preprocess,remove_stopwords
from model.attention import *
from model.decoder import *
from model.encoder import *
from model.seq2seq import *



import torch


src_path = '/home/dhurba/Documents/smart_write/deployment/src.pth'
trg_path = '/home/dhurba/Documents/smart_write/deployment/trg.pth'

SRC = torch.load(src_path, map_location=torch.device('cpu'))
TRG = torch.load(trg_path, map_location=torch.device('cpu'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)


ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
ENC_HID_DIM = 512
DEC_HID_DIM = 512

ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)


model_path='/home/dhurba/Documents/smart_write/deployment/smart_write_5.pt'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


def predict_phrase(sentence, src_field, trg_field, model, device, max_len = 5):

    model.eval()
        
    #tokenize the source sentence
    if type(sentence)==str:    #if input phrase is a string (it is needed in inference)
        tokens=sentence_into_words(sentence)
    else:                       #if input phrase is a list of word (it is needed in bleu score calculation)
        input_phrase_generated=' '.join(sentence)
        tokens=sentence_into_words(input_phrase_generated)

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)
        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention
            
        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:-1], attentions[:len(trg_tokens)-1]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        context_phrase = request.form['context_phrase']

        context_phrase=preprocess(context_phrase)
        context_phrase=remove_stopwords(context_phrase)

        predcited_phrase,_=predict_phrase(context_phrase, SRC, TRG, model,device)

        predcited_phrase=' '.join(predcited_phrase)

    return render_template('result.html', prediction=predcited_phrase)

if __name__ == '__main__':
    app.run(debug=True)


