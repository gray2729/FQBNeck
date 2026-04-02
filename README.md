# FQBNeck: Detecting AI Images using Hybrid Approach of Frequency Analysis and Information Bottlenecks

This project aims to distinguish between real and AI-generated images using a multi-stream image classification framework that combines RGB spatial information and hidden frequency patterns revealed by FFT. This framework further leverages a variational information bottleneck to improve generalization across GANs and DM generated images.

**Author:** Isaac Gray **College:** Rochester Institute of Technology **Email:** ig8317@g.rit.edu

## Model Framework

## Requirements

* python = 3.10 
* numpy = 2.0.1 
* pyyaml = 6.0.3 
* torch = 2.5.1 
* torchvision = 0.20.1 
* pillow = 11.1.0 
* scikit-learn = 1.7.1 
* matplotlib = 3.10.8 
* pandas = 2.3.3
* pytorch-cuda = 12.1

## Install

1. Clone the repository and navigate to the FQBNeck folder
```
https://github.com/gray2729/FQBNeck
cd FQBNeck
```
2. Install requirements: Create conda  environment
```
conda create -n  FQBNeck_env python-3.10 -y
conda activate FQBNeck_env
pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

```
FQBNeck/
    ├── datasets/         # Dataset storage
    │   ├── Hybrid/                 # Individual dataset
    |   |   ├── Training/
    |   |   ├── Validation/
    |   |   └── Testing/
    │   └── ...
    ├── figures/          # Images of datasets and results
    │   ├── dataset_distributions/  # Data distributions in dataset
    │   ├── dataset_samples/        # Image samples from dataset 
    │   ├── loss_curves/            # Training loss curves during Training
    │   └── ..
    ├── results/          # Testing Results
    │   ├── FQBNeck_Hybrid/         # Training/Testing results for individual model
    |   |   ├── config.yaml            # Config used for Training  
    |   |   ├── losses.csv             # Recorded losses during Training
    |   |   └── metrics.json           # Recorded metrics during Testing 
    │   └── ..
    ├── saved_models/     # Trained models
    │   ├── FQBNeck_Hybrid.pt       # Saved individual model
    │   └── ..
    ├── scripts/          # Scripts
    │   ├── configs/                # Configs for Training
    |   |   ├── configs.yaml
    |   |   └── ...
    │   ├── data/                   # Image data loading scripts
    │   ├── models/                 # Model architecture scripts
    │   ├── training/               # Training/Validation/Testing loop scripts
    │   └── utils/                  # Utility scripts
    ├── main.py           # Script for running Training/Testing loop
    ├── baselines.py      # Script for running baseline
    ├── requirements.txt  # Project requirements
    ├── README.md         # Project documentation
    └── LICENSE           # License 
```

## Usage

## Demo

Run main.py script with the following command-line arguments to train a model on the specific demo dataset (Hybrid_Sample) using the demo configs (sample_configs):
```
python main.py --dataset Hybrid_Sample --config sample_configs --process training --model_name fbqneck_demo
``` 

This will train a model for a specified number of epochs listed in the configs before testing the model.

Likewise, run main.py script with the following command-line arguments to test a model (FQBNeck) on a specific demo dataset (Hybrid_Sample) using the demo configs (sample_configs):

```
python main.py --dataset Hybrid_Sample --config sample_configs --process testing --model_name fqbneck_hybrid
```

This will print the metrics out in the terminal as well as save the metrics as metrics.json in results folder within the corresponding model folder.

You can test any model trained by replacing the model name in the command-line arguments above with the desired model. Note that the desired model has to be in the saved_dataset folder. For example, to test model trained for the demo, replace fqbneck_hybrid above with fbqneck_demo as such:

```
python main.py --dataset Hybrid_Sample --config sample_configs --process testing --model_name fbqneck_demo
```

## Results and Visualization

## Citation

If you use this code in your research, please cite:

```
@article{
    gray,
    title="FQBNeck: Detecting AI Images using Hybrid Approach of Frequency Analysis and Information Bottlenecks",
    author="Isaac Gray",
    institution="Rochester Institute of Technology",
    year="2026"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/gray2729/FQBNeck/blob/main/LICENSE) file for details.