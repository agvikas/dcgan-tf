# dcgan-tf
Implementation of Deep Convolutional Generative Adversarial Network in tensorflow. For original paper click [here](https://arxiv.org/pdf/1511.06434.pdf).
The dataset used to train was a custom dataset containing paintings of 45 artists. Wide genre of paintings ranging from abstract to portarit were included.

## Requirements
All requirements can be installed by running ```pip install -r requirements.txt ```. If you want to manually install, the list is given below.  
- python 2.7 / 3.5  
- tensorflow-gpu  
- numpy  
- opencv  

## Usage
```python main.py ```  
or
'''python main.py --parameters <path_to_parameters_file_in_JSON_format> --mode <True_or_False_for_training_or_inference>

#### arguments:

| Argument      | default  | Description  |
| ------------- |:------------|:------------|
| --parameters   | "parameters.json" | Path to file containing parameters in JSON format|
| --mode | True          | True for training or False for inference |

## Sample Outputs
Genreted images during training are shown below.

After 15<sup>th</sup> epoch:

<img src="https://user-images.githubusercontent.com/38666732/49446049-4b79f000-f7f9-11e8-8b0c-2e26a8d1e220.png" width='500'>


After 207<sup>th</sup> epoch:

<img src="https://user-images.githubusercontent.com/38666732/49446749-e2937780-f7fa-11e8-95e3-7dc0a921e97b.png" width='500'>

After 334<sup>th</sup> epoch:

<img src="https://user-images.githubusercontent.com/38666732/49446932-43bb4b00-f7fb-11e8-9681-4d58fc6bb483.png" width='500'>





