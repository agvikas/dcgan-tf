# dcgan-tf
Implementation of Deep Convolutional Generative Adversarial Network in tensorflow. For original paper click [here](https://arxiv.org/pdf/1511.06434.pdf).
The dataset used to train was a custom dataset containing portrait paintings of 45 artists.

## Requirements
All requirements can be installed by running ```pip install -r requirements.txt ```. If you want to manually install, the list is given below.  
- python 2.7 / 3.5  
- tensorflow-gpu  
- numpy  
- opencv  

## Usage
```python main.py ```  
or
```python main.py --parameters <path_to_parameters_file_in_JSON_format> --mode <True_or_False_for_training_or_inference>```

#### arguments:

| Argument      | default  | Description  |
| ------------- |:------------|:------------|
| --parameters   | "parameters.json" | Path to file containing parameters in JSON format|
| --mode | True          | True for training or False for inference |

## Sample Outputs
Some of the genreted images during training are shown in gif below.

![](/gen_images/movie.gif)

Images generated post training for 510 epochs:

![gen_inf-48](https://user-images.githubusercontent.com/38666732/51323901-69d93980-1a8f-11e9-95e5-8272c46f3833.jpg)
![gen_inf-15](https://user-images.githubusercontent.com/38666732/51323905-6b0a6680-1a8f-11e9-8411-0876dd6f34e0.jpg)
![gen_inf-53](https://user-images.githubusercontent.com/38666732/51323909-6d6cc080-1a8f-11e9-8282-1864e02c85c1.jpg)
![gen_inf-46](https://user-images.githubusercontent.com/38666732/51323923-75c4fb80-1a8f-11e9-83ce-4ea376c844d5.jpg)
![gen_inf-50](https://user-images.githubusercontent.com/38666732/51323931-78275580-1a8f-11e9-9007-da39081f4526.jpg)
![gen_inf-48-1](https://user-images.githubusercontent.com/38666732/51324094-d2c0b180-1a8f-11e9-91d5-7f09171b22ee.jpg)
![gen_inf-38](https://user-images.githubusercontent.com/38666732/51324096-d5230b80-1a8f-11e9-94d0-d7b215bcc302.jpg)
![gen_inf-55](https://user-images.githubusercontent.com/38666732/51324195-13b8c600-1a90-11e9-9fdd-cc182e2dedb1.jpg)











