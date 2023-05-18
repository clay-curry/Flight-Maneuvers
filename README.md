<div align="center">

<h1>Steerable Inductive Biases for Flight Maneuver Reconstruction</h1>

<h4><b>Clayton Curry</b></h4>

<br>

[![Blog: Flight Maneuvers](https://img.shields.io/badge/Blog-Flight%20Maneuvers-blue)](https://claycurry.com/blog/maneuver)
[![Flask Project Demo](https://img.shields.io/badge/Flask-Project%20Demo-ff69b4)](https://claycurry.com/blog/maneuvers)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13U6SEHBlYxXKCvmLBXeERzIb5dt2lEfA?usp=sharing)
[![Github Repo](https://img.shields.io/static/v1.svg?logo=github&label=repo&message=see%20project&color=blueviolet)](https://github.com/clay-curry/Flight-Maneuvers) 

</div>


## Description

TODO

### Related Projects

* [escnn](https://github.com/QUVA-Lab/escnn)
* [e3nn](https://github.com/e3nn/e3nn/)
* [geometric gnn dojo](https://github.com/chaitjo/geometric-gnn-dojo)
* [e3nn-jax](https://github.com/e3nn/e3nn-jax)
* [equivariant-MLP](https://github.com/mfinzi/equivariant-MLP)
* [SignNet-BasisNet](https://github.com/cptq/SignNet-BasisNet)
* [mace-jax](https://github.com/ACEsuit/mace-jax)

## Directory Structure and Usage

```
.
├── README.md
├── app.py                              # opens web app on localhost created using Dash and Flask 
├── main.py                             # runs experiments on local hardware after parsing CLI arguments
│
├── examples
│   ├── introduction.ipynb              # A gentle introduction to background and methodology used in project
│   └── TODO                            # TODO
│
│
└── src
    ├── model
    │   ├── resnet_1D.py                # ordinary residual blocks and network operating on a sampled timeseries
    │   ├── se2_resnet_1D.py            # TODO
    │   └── se3_resnet_1D.py            # TODO
    │
    ├── data_module.py                  # defines classes extending pytorch data utilities 
    ├── neural_search.py                # TODO
    └── utils.py                        # standard utilities for plotting results, obtaining statistics, and pre-/postprocessing data
```



## How to run

First, create a new conda environment *using Python 3.7* and activate it as follows

```bash
# create env
conda create -n flight-maneuvers python=3.7
# activate it
conda activate flight-maneuvers
 ```   

Next, clone the repo, navigate to the project base, and install requirements,
 ```bash
# clone project   
git clone https://github.com/clay-curry/Flight-Maneuvers
# enter project
cd Flight-Faneuvers
# install dependencies
pip install -r requirements.txt
```

Finally, pass a supporting argument to the main module as follows
```
# run module (example: mnist as your main contribution)   
python main.py  COMMAND
```

where `COMMAND` is chosen from the following table:

| COMMAND  | Description |
| -------- | ----------- |
| fetch    | invokes 'curl' to download 1000 pre-simulated training examples |
| train    | instantiates a model and invokes routines for training |
| evaluate | assesses model performance using a randomized ANOVA procedure |
| app      | starts a web server on localhost and serves models using Flask |