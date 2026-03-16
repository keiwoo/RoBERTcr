import torch, sys, os, shutil
import pandas as pd
import numpy as np
from transformers import (
    BertTokenizer,
    RobertaModel,
    Trainer,
    TrainingArguments,
    RobertaConfig
)
from torch.utils.data import Dataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)

torch.manual_seed(301)
np.random.seed(301)
import time


tokenizer_kwargs = {
    'padding': 'max_length',
    'truncation': True,
}

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

class myDataset(Dataset):
    def __init__(self, file):
        self.tcrs = pd.read_csv(file)['tcr'].values.tolist()
        self.peps = pd.read_csv(file)['peptide'].values.tolist()
        self.labels = pd.read_csv(file)['label'].values.tolist()
        self.tokenizer1 = BertTokenizer.from_pretrained(
            './pre-trained-model/tcr',
            do_lower_case = False
        )
        self.tokenizer2 = BertTokenizer.from_pretrained(
            './pre-trained-model/peptide', 
            do_lower_case = False
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        tcr = self.tcrs[item]
        pep = self.peps[item]
        label = self.labels[item]
        tokenized1 = self.tokenizer1(
            ' '.join(tcr), 
            max_length=30,
            **tokenizer_kwargs
        )
        tokenized2 = self.tokenizer2(
            ' '.join(pep),
            max_length=14,
            **tokenizer_kwargs
        )
        tokenizedData = {
            'input_ids1': tokenized1['input_ids'],
            'attention_mask1': tokenized1['attention_mask'],
            'token_type_ids1': tokenized1['token_type_ids'],
            'input_ids2': tokenized2['input_ids'],
            'attention_mask2': tokenized2['attention_mask'],
            'token_type_ids2': tokenized2['token_type_ids'],
            'labels': torch.tensor(label, dtype=torch.float32),
        }
        return tokenizedData
def get_mean_pooled_output(hidden_state, attention_mask):
    mask = attention_mask.bool()
    mask[:, 0] = False
    
    seq_lengths = attention_mask.sum(dim=1)
    batch_indices = torch.arange(
        hidden_state.size(0), 
        device=hidden_state.device
    )
    mask[batch_indices, seq_lengths - 1] = False
    
    mask = mask.unsqueeze(-1).expand_as(hidden_state)

    return (hidden_state * mask).sum(dim=1) / (mask.sum(dim=1))
class myModel(RobertaModel):
    def __init__(self, config):
        super(myModel, self).__init__(config)
        self.roberta1 = RobertaModel.from_pretrained('./pre-trained-model/tcr')
        self.roberta2 = RobertaModel.from_pretrained('./pre-trained-model/peptide')
        self.ln = torch.nn.Linear(1024*3, 1024*3)
        self.classifier = torch.nn.Linear(1024*3, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.loss_fct = torch.nn.BCEWithLogitsLoss()
    def forward(self, 
                input_ids1,
                attention_mask1,
                token_type_ids1,
                input_ids2,
                attention_mask2,
                token_type_ids2,
                labels=None
        ):
        
        output1 = self.roberta1(
            input_ids1, 
            attention_mask1, 
            token_type_ids1, 
            output_hidden_states=True
        ).last_hidden_state
        output1 =get_mean_pooled_output(output1, attention_mask1)

        output2 = self.roberta2(
            input_ids2, 
            attention_mask2, 
            token_type_ids2, 
            output_hidden_states=True
        ).last_hidden_state
        output2 =get_mean_pooled_output(output2, attention_mask2)

        output = torch.cat(
            (
            output1,
            output2, 
            torch.mul(output1, output2)
            ),
            dim=1
        )
            
        logits = self.classifier(
            self.dropout(
                torch.nn.functional.relu(
                    self.ln(output)
                )
            )
        )
        loss = self.loss_fct(logits.view(-1), labels.view(-1))
        return loss, logits
    
def compute_metric(pred):
    logits, labels = pred
    preds = 1/(1 + np.exp(-logits))
    # pd.DataFrame({
    #     'labels':labels.flatten(),
    #     'preds':preds.flatten()
    # }).to_csv(
    #     f'{output_dir}/result_{time.time()}.csv',
    #     index=False
    # )

    return {
        'roc': roc_auc_score(labels, preds),
        'ap': average_precision_score(labels, preds),
        'acc': accuracy_score(labels, preds.round()),
        'f1': f1_score(labels, preds.round()),
        're': recall_score(labels, preds.round()),
        'pr': precision_score(labels, preds.round()),
    }



output_dir = f'tmp{sys.argv[1]}'
train_dataset = myDataset(f'/data/kwzhou/RoBERTcr/TRAP_full_random_train.csv' )
test_dataset = myDataset(f'/data/kwzhou/RoBERTcr/datasets/TRAP/randomly-test.csv')
config = RobertaConfig.from_pretrained('./pre-trained-model/tcr')
model = myModel(config)
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="best",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="roc",
    greater_is_better=True,
    learning_rate=0.00005,
    overwrite_output_dir=True,
    bf16=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metric,
)
trainer.train()
print(trainer.evaluate())
