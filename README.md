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
 
# Llmm Repository

This repository contains the implementation of Llmm, which utilizes LoRA and LoftQ to optimize parameter distributions in the probability measure space, enhancing efficiency and effectiveness in natural language tasks.

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

For specific parameter configurations, please refer to the accompanying paper.

### Training Steps

1. **Generate Quantized Models**:  
   Run the following script to generate quantized models:
   ```bash
   bash _init.sh
   ```

2. **Train Llama and Llama2 on Alpaca and Hh-rlhf Datasets**:  
   Use the script below:
   ```bash
   bash train_nlg_llama.sh
   ```

3. **Train and Evaluate Llama2 on gsm8k Dataset**:  
   - To train:
     ```bash
     bash train_gsm8k.sh
     ```
   - To evaluate:
     ```bash
     bash test_gsm8k.sh
     ```

4. **Train DeBERTaV3-base on GLUE**:  
   Execute the following script:
   ```bash
   bash train_glue_llmm.sh
   ```

### Scale Size Configuration

- **Steps 2 & 3**: Scale size `$s` is configured in `./src/loramodel.py`.
- **Step 4**: Scale size is set in `./utils_llmm.py`.

---

## Additional Information

For detailed descriptions and theoretical background, refer to the paper provided with this repository.

