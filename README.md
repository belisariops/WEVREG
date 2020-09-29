## Install

#### Miniconda

[Miniconda latest installer](https://docs.conda.io/en/latest/miniconda.html)

Add mininconda to your PATH:

```
$ bash <installer_directory>/Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda 
$ echo "PATH=$PATH:$HOME/miniconda/bin" >> .bashrc
$ source .bashrc
$ conda -V
conda 4.6.7
```



Load environment:
``
$ conda env create --file=environment.yml
``


Run examples:

```
$ cd <Project-DIR>
$ source activate WEVREG
$ python -m experiments.feature_selection
```



