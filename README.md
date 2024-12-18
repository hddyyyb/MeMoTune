## Environment
- **Python**: 3.10  
- **CUDA Toolkit**: 11.8  
- **Transformers**: 4.41.2  
- **PyTorch**: 2.1.2+cu118  
- **PEFT**: latest version  
- **TensorFlow-GPU**: 2.14.0  
- **cuDNN**: 8.9.2.26  

## Parameter Distribution Training
The parameter distribution training files are located in `./src` and will overwrite the corresponding library files.  

The proposed parameter distribution update method is primarily implemented in:  
- `./src/loramodel.py`  
- `./src/loralayer.py`  

For specific parameter configurations, please refer to the paper.
 
1. Run _init.sh to generate quantized models.

2. Run train_nlg_llama.sh to train Llama and Llama2 on dataset Alpaca and Hh-rlhf. 

3. Run train_gsm8k.sh to train Llama2 on dataset gsm8k, and run test_gsm8k.sh to evaluate the trained models. 

4. Run train_glue_llmm.sh to train DeBERTaV3-base on GLUE.

scale size $s$ of 2&3 is configured at ./src/loramodel.py, while for 4, it is set in ./utils_llmm.py 
