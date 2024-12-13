envirenment: python==3.10,cudatoolkit==11.8,transformers==4.41.2,torch==2.1.2+cu118,peft,tensorflow-gpu==2.14.0,cudnn==8.9.2.26

The parameter distribution training files are located in ./src and will replace the corresponding library files. The proposed parameter distribution update method is primarily implemented in ./src/loramodel.py and ./src/loralayer.py.

1. Run _init.sh to generate quantized models.

2. Run train_nlg_llama.sh to train Llama and Llama2 on dataset Alpaca and Hh-rlhf. 

   | learning rate | llmmscale | max_steps | K    |
   | ------------- | --------- | --------- | ---- |
   | 1e-2          | 400       | 30000     | 2    |

3. Run train_gsm8k.sh to train Llama2 on dataset gsm8k, and run test_gsm8k.sh to evaluate the trained models. 

   | learning rate | llmmscale | max_steps | K    |
   | ------------- | --------- | --------- | ---- |
   | 3e-4          | 400       | 1880      | 2    |

4. Run train_glue_llmm.sh to train DeBERTaV3-base on GLUE.

   The hyperparameters that need to be modified are as follows: 

   | quant        | rank | dataset | learning rate | llmmscale | epoch | K    |
   | ------------ | ---- | ------- | ------------- | --------- | ----- | ---- |
   | normal_float | 32   | cola    | 5e-5          | 600       | 60    | 1    |
   | normal_float | 32   | mrpc    | 1e-4          | 600       | 60    | 1    |
   | normal_float | 32   | stsb    | 1e-4          | 700       | 60    | 1    |
   | normal_float | 16   | cola    | 1e-4          | 1000      | 60    | 1    |
   | normal_float | 16   | mrpc    | 1e-4          | 500       | 60    | 1    |
   | normal_float | 16   | stsb    | 1e-4          | 700       | 60    | 1    |
   | uniform      | 32   | cola    | 1e-4          | 4000      | 60    | 1    |
   | uniform      | 32   | mrpc    | 1e-4          | 100       | 60    | 1    |
   | uniform      | 32   | stsb    | 5e-5          | 300       | 60    | 1    |
   | uniform      | 16   | cola    | 1e-4          | 1000      | 60    | 2    |
   | uniform      | 16   | mrpc    | 1e-4          | 700       | 60    | 1    |
   | uniform      | 16   | stsb    | 5e-5          | 300       | 60    | 1    |

    (llmmscale of 2&3 is configured at ./src/loramodel.py, while for 4, it is set in ./utils_llmm.py)
