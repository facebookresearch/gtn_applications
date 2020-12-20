# gtn_applications

An applications library using [GTN](https://github.com/facebookresearch/gtn).
Current examples include:

- Offline handwriting recognition
- Automatic speech recognition

## Installing

1. Build python bindings for the [GTN library](https://github.com/fairinternal/gtn#using-python-bindings).

2. `conda activate gtn_env # using the same environment from Step 1`

3. `conda install pytorch torchvision -c pytorch`

4. `pip install -r requirements.txt`

## Training

We give an example of how to train on the [IAM off-line handwriting recognition](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
benchmark. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/gtn/blob/master/examples/notebooks/IAM_Handwriting_Recognition.ipynb)

First register [here](https://fki.tic.heia-fr.ch/login) and download the dataset:
```
./datasets/download/iamdb.sh <path_to_data> <email> <password>
```

Then update the configuration JSON `configs/iamdb/tds2d.json` to point to the
data path used above:
```
  "data" : {
    "dataset" : "iamdb",
    "data_path" : "<path_to_data>",
    "num_features" : 64
  },
```

Single GPU training can be run with:
```
python train.py --config configs/iamdb/tds2d.json
```

To run distributed training with multiple GPUs:
```
python train.py --config configs/iamdb/tds2d.json --world_size <NUM_GPUS>
```

For a list of options type:
```
python train.py -h
```

## Contributing

Use [Black](https://github.com/psf/black) to format python code.

First install:

```
pip install black
```

Then run with:

```
black <file>.py
```
### License

GTN is licensed under a MIT license. See [LICENSE](LICENSE).
