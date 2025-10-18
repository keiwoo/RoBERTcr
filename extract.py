from transformers import (
    BertTokenizer,
    RobertaModel,
)

import torch, pickle


input_file = "sequences.txt"
model_name = "/data/kwzhou/RoBERTcr/pre-trained-model/tcr" 
mean = False

tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = False)
model = RobertaModel.from_pretrained(model_name)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

with open(input_file, "r") as f:
    sequences = [line.strip() for line in f if line.strip()]

results = {}
with torch.no_grad():
    if mean:
        for idx, seq in enumerate(sequences):
            outputs = model(**tokenizer(' '.join(seq), return_tensors="pt").to(device)).last_hidden_state.squeeze(0)[1:-1]
            results[seq] = torch.mean(outputs, dim=0).cpu()
            print(f"\rProcessed: {idx+1}/{len(sequences)}", end="")
    else:
        for idx, seq in enumerate(sequences):
            outputs = model(**tokenizer(' '.join(seq), return_tensors="pt").to(device)).last_hidden_state.squeeze(0)[1:-1]
            results[seq] = outputs.cpu()
            print(f"\rProcessed: {idx+1}/{len(sequences)}", end="")
print()
with open(f"{input_file.split('.')[0]}.pkl", "wb") as f:
    pickle.dump(results, f)
