# rightrec

A library for handwriting recognition with GTNs.

## Installing

1. Build python bindings for the [GTN library](https://github.com/fairinternal/gtn#using-python-bindings).

2. `conda activate gtn_env # using the same environment from Step 1`

3. `conda install pytorch torchvision -c pytorch`

4. `pip install -r requirements.txt`

## Training

We give an example of how to trian on the [IAM off-line handwriting recognition](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
benchmark.

First download the dataset:
```
./datasets/download_iamdb.sh <path_to_data>
```

Then update the configuration JSON `configs/iamdb/tds2d.json` to point to the
data path used above:
```
  "data" : {
    "dataset" : "iamdb",
    "data_path" : "<path_to_data>",
    "img_height" : 64
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
