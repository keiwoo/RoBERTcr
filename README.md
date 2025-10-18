This repository contains relevant information of our research article *Supervised fine-tuning enhances unsupervised learning from 30 million amino acids in TCR sequences*

## Reproduce results quickly
We already provide our prediction results of every dataset in their corresponding file folder. If you want to train and test the model by yourself, please follow the instructions below.
### 1. Download models

The pre-trained models and its datasets are at [Hugging Face](https://huggingface.co/keiwoo), you should download and put them in right place. Please follow the instructions in every model file folder.

### 2. Setup running environment
Though the main Python packages we use are
```
python=3.12.7
transformers=4.51.1
torch=2.6.0
numpy=2.2.4
pandas=2.2.3
cuda=12.4
```
There is no feature or function needing specific version of package, you can run `RoBERTcr.py` in your existing Python environment. 

> If your device does not support `BF16` precision mode (GPU based on NVIDIA's Ampere and its subsequent architectures), you should set `bf16=False` at [line 184](./RoBERTcr.py#L184) which will cost 4x time more then at most.

### 3. Run the code

Change the dataset path at [line 165](./RoBERTcr.py#L165) and [line 166](./RoBERTcr.py#L166). The format of dataset is CSV file separated by comma. The CSV header should contain `label`, `tcr` and `peptide` that no need to be in order. Run `python RoBERTcr.py your_gpu_id`. If you want to save prediction result, uncomment [line 145-151](./RoBERTcr.py#L145-L151) and it will save every epoch.

## Reuse model encoding

We provide a script for extracting pre-trained embedding conveniently. This script will save the embedding as dictionary in a pickle file, where the key is the sequence and the value is its embedding. The instructions are as follows:

1. Put your sequences in a TEXT file line by line. 
2. Change `your_sequences.txt` file path at [line 9](./extract.py#L9)
3. Change model at [line 10](./extract.py#L10).
4. Set `mean=True` at [line 11](./extract.py#L11) if you want to use mean pooling to get fixed-length embedding.
5. Change your device at [line 15](./extract.py#L15).
6. Run the script with `python extract.py` and it will save the embedding in the same directory as `your_sequences.txt` with the name `your_sequences.pkl`.

## Web server
We also provide a web server for users to test their TCR-peptide pairs without running code. The web server is at [RoBERTcr Web Server](http:).

## Acknowledgements
We express our thanks to these researches.