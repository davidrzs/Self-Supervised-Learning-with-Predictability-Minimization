# Code Accompanying the Paper

## Setting up the Environment

Use the provided Pipfile to set up the environment:

```bash
pipenv install
```

and then to activate the environment use:

```bash
pipenv shell
```

To log our runs we use Weights & Biases, make sure you have an account and are locally authenticated.

## To Train Models

For **CLPM-Reg**, use `cl_lin_pred_min.yaml` and train as follows:

```bash
python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min.yaml
```

For **CLPM-Opt**, use `cl_non_lin_pred_minv6.yaml` for the 3 layer predictor and `cl_non_lin_pred_minv6_1layer.yaml` for the 1 layer predictor

```bash
python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_non_lin_pred_minv6.yaml
```

```bash
python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_non_lin_pred_minv6_1layer.yaml
```

For CIFAR-10 and CIFAR-100 we have used the same files within the `scripts/pretrain/cifar/` folder whereas for ImageNet-100 we have used the files within the `scripts/pretrain/imagenet-100/` folder.

Go into the YAML files and set the data paths accordingly! The results in terms of accuracy and losses should then be visible on Weights & Biases.

## Linear Evaluation of ImageNet-100

We follow the solo-learn library and use the following 100 categories of the original ImageNet dataset: [ImageNet-100 categories](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt)

Use the following command but exchange the `pretrained_feature_extractor` and the data paths. You can leave the `--config-name` as `cl_lin_pred_min` 

```bash
python main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml pretrained_feature_extractor="./trained_models/cl_lin_pred_min/96797f2p/cl_lin_pred_min-imagenet100-96797f2p-ep\=399.ckpt" data.train_path="./ILSVRC2012_img_train_100/" data.val_path="./ILSVRC2012_img_val_100/"
```

## To Extract Embeddings from Trained Models

Run the file `bash ems_corr_test.sh` and set the weights and biases IDs of the runs you would like to extract the embeddings from. The embeddings will be placed in the folder `correlation_analysis`.

## Get Redundancy

```bash
python analysis/predict_all_refactored_new.py --folder ./correlation_analysis/ --subsample_rate 0.2
```

This will output an Excel file that contains all redundancy measures for the extracted embedding from where the redundancy measures can be read off. 