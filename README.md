 # Llmm Repository

This repository contains the implementation of Llmm, enhancing efficiency and effectiveness in natural language tasks.

---

## Environment

Ensure the following dependencies are installed:

- **Python**: 3.10  
- **CUDA Toolkit**: 11.8  
- **Transformers**: 4.41.2  
- **PyTorch**: 2.1.2+cu118  
- **PEFT**: Latest version  
- **TensorFlow-GPU**: 2.14.0  
- **cuDNN**: 8.9.2.26  

---

## Parameter Distribution Training

### Overview
The parameter distribution training files are located in the `./src` directory and will overwrite the corresponding library files. The key implementation files for the proposed parameter distribution update method are:

- `./src/loramodel.py`
- `./src/loralayer.py`

For specific parameter configurations, please refer to the paper.
 

### Training Steps

1. **Generate Quantized Models**:  
   Run `_init.sh` to generate quantized models.
   
2. **Train Llama and Llama2 on Alpaca and Hh-rlhf Datasets**:  
   Run `train_nlg_llama.sh` to train Llama and Llama2 on dataset Alpaca and Hh-rlhf. 

3. **Train and Evaluate Llama2 on gsm8k Dataset**:  
   Run `train_gsm8k.sh` to train Llama2 on dataset gsm8k, and run `test_gsm8k.sh` to evaluate the trained models. 

4. **Train DeBERTaV3-base on GLUE**:  
   Run `train_glue_llmm.sh` to train DeBERTaV3-base on GLUE.


---

## Additional Information

For detailed descriptions and theoretical background, refer to the paper provided with this repository.

  
