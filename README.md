# ComputationalSemantics
Computationsl Semantics Assignments UPF

For Assignement 2, to run dilbert, you will need to download [this](https://huggingface.co/visualjoyce/transformers4vl-vilbert).

Do the following:

```
!apt-get install git-lfs
!git-lfs install
!git clone https://huggingface.co/visualjoyce/transformers4vl-vilbert
!rm /content/transformers4vl-vilbert/config.json
!mv /content/transformers4vl-vilbert/bert_base_6layer_6conect.json /content/transformers4vl-vilbert/config.json
```

And to load the model use appropriate paths.
```
from transformers import AutoModel
model = AutoModel.from_pretrained("/content/transformers4vl-vilbert",output_hidden_states=True)
```

The above snippet is designed for Colab, and with suitable tweaks will work on local machine. 

### For Assignment 3

Check [CommentOnly](https://github.com/aakash94/ComputationalSemantics/tree/CommentOnly) branch for best performance.
Make sure following libraries are installed.
- Pandas
- Numpy
- Torch
- HuggingFace
- HappyTransformer
- Sklearn
- Seaborn
- Matplotlib