import argparse
from dataclasses import dataclass
import editdistance
import json
import logging
import os
import sys
import time
import torch

import datasets
import models
import utils
import transducer


def parse_args():
    parser = argparse.ArgumentParser(
        description="IAM Handwriting Recognition with Pytorch.")
    parser.add_argument("--config",
                        type=str,
                        help="A json configuration file for experiment.")
    parser.add_argument("--use_gtn", action="store_true", help="Use GTN")
    parser.add_argument(
        "--checkpoint_path",
        default="/tmp/",
        type=str,
        help="Checkpoint path for saving models",
    )
    parser.add_argument("--world_size",
                        default=1,
                        type=int,
                        help="world size for distributed training")
    parser.add_argument(
        "--dist_url",
        default="tcp://localhost:23146",
        type=str,
        help="url used to set up distributed training. This should be"
        "the IP address and open port number of the master node",
    )
    parser.add_argument("--dist_backend",
                        default="nccl",
                        type=str,
                        help="distributed backend")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not torch.cuda.is_available():
        logging.fatal("No available CUDA devices")
        sys.exit(1)

    logging.info(("" if args.use_gtn else "Not ") + "Using GTN")

    logging.info("World size is : " + str(args.world_size))

    if torch.cuda.device_count() < args.world_size:
        logging.fatal("At least {} cuda devices required. {} found".format(
            args.world_size, torch.cuda.device_count()))
        sys.exit(1)

    return args


def compute_edit_distance(predictions, targets):
    dist = 0
    n_tokens = 0
    for p, t in zip(predictions, targets):
        dist += editdistance.eval(p.tolist(), t.tolist())
        n_tokens += t.numel()
    return dist, n_tokens


@dataclass
class Meters:
    loss = 0.0
    num_samples = 0
    num_tokens = 0
    edit_distance = 0

    def sync(self):
        lst = [
            self.loss, self.num_samples, self.num_tokens, self.edit_distance
        ]
        # TODO: avoid this so that cpu training also works
        lst_tensor = torch.FloatTensor(lst).cuda()
        torch.distributed.all_reduce(lst_tensor)
        (
            self.loss,
            self.num_samples,
            self.num_tokens,
            self.edit_distance,
        ) = lst_tensor.tolist()

    @property
    def avg_loss(self):
        return self.loss / self.num_samples

    @property
    def cer(self):
        return self.edit_distance / self.num_tokens


@torch.no_grad()
def test(model, criterion, data_loader, world_rank, world_size):
    model.eval()
    meters = Meters()
    for inputs, targets in data_loader:
        outputs = model(inputs.to(world_rank))
        meters.loss += criterion(outputs, targets).item() * len(targets)
        meters.num_samples += len(targets)
        dist, toks = compute_edit_distance(criterion.decode(outputs), targets)
        meters.edit_distance += dist
        meters.num_tokens += toks
    if world_size > 1:
        meters.sync()
    return meters.avg_loss, meters.cer


def checkpoint(model, criterion, checkpoint_path, save_best=False):
    model_checkpoint = os.path.join(checkpoint_path, "model.checkpoint")
    criterion_checkpoint = os.path.join(checkpoint_path,
                                        "criterion.checkpoint")
    torch.save(model.state_dict(), model_checkpoint)
    torch.save(criterion.state_dict(), criterion_checkpoint)
    if save_best:
        torch.save(model.state_dict(), model_checkpoint + ".best")
        torch.save(criterion.state_dict(), criterion_checkpoint + ".best")


def train(world_rank, args):
    # setup logging
    level = logging.INFO
    if world_rank != 0:
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    with open(args.config, "r") as fid:
        config = json.load(fid)
    is_distributed_train = False
    if args.world_size > 1:
        is_distributed_train = True
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=world_rank,
        )

    # TODO: avoid this so that cpu training also works
    torch.cuda.set_device(world_rank)

    # seed everything:
    seed = config.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)

    # setup data loaders:
    dataset = config["data"]["dataset"]
    if not (hasattr(datasets, dataset)):
        raise ValueError(f"Unknown dataset {dataset}")
    dataset = getattr(datasets, dataset)

    input_size = config["data"]["img_height"]
    data_path = config["data"]["data_path"]
    preprocessor = dataset.Preprocessor(
            data_path,
            img_height=input_size,
            tokens_path=config["data"].get("tokens_path"))
    import pdb
    pdb.set_trace()
    # TODO update preprocessor to load tokens from file is specified
    trainset = dataset.Dataset(data_path,
                               preprocessor,
                               split="train",
                               augment=True)
    valset = dataset.Dataset(data_path, preprocessor, split="validation")
    train_loader = utils.data_loader(trainset, config, world_rank,
                                     args.world_size)
    val_loader = utils.data_loader(valset, config, world_rank, args.world_size)

    # setup Model:
    output_size = preprocessor.num_classes
    if config["criterion"]["blank"]:
         output_size += 1  # account for blank
    model = models.load_model(config["model_type"], input_size, output_size,
                              config["model"]).to(world_rank)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info("Training {} model with {:,} parameters.".format(
        config["model_type"], n_params))

    if args.use_gtn:
        criterion = transducer.Transducer(preprocessor.tokens, preprocess.graphemes)
    else:
        criterion = models.CTC(blank=output_size - 1,
                               use_gtn=args.use_gtn).to(world_rank)

    if is_distributed_train:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[world_rank])
        if len(list(criterion.parameters())) > 0:
            criterion = torch.nn.parallel.DistributedDataParallel(
                criterion, device_ids=[world_rank])

    epochs = config["optim"]["epochs"]
    lr = config["optim"]["learning_rate"]
    step_size = config["optim"]["step_size"]
    max_grad_norm = config["optim"].get("max_grad_norm", None)

    # run training:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=step_size,
                                                gamma=0.5)

    min_val_loss = float("inf")
    min_val_cer = float("inf")

    timers = utils.Timer([
        "ds_fetch",  # dataset sample fetch
        "model_fwd",  # model forward
        "crit_fwd",  # criterion forward
        "bwd",  # backward (model + criterion)
        "optim",  # optimizer step
        "metrics",  # decode, cer
        "train_total",  # total training
        "test_total",  # total testing
    ])
    num_updates = 0
    for epoch in range(epochs):
        model.train()
        criterion.train()
        start_time = time.time()
        meters = Meters()
        timers.reset()
        timers.start("train_total").start("ds_fetch")
        for inputs, targets in train_loader:
            timers.stop("ds_fetch").start("model_fwd")
            optimizer.zero_grad()
            outputs = model(inputs.to(world_rank))
            timers.stop("model_fwd").start("crit_fwd")
            loss = criterion(outputs, targets)
            timers.stop("crit_fwd").start("bwd")
            loss.backward()
            timers.stop("bwd").start("optim")
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_grad_norm)
            optimizer.step()
            num_updates += 1
            timers.stop("optim").start("metrics")
            meters.loss += loss.item() * len(targets)
            meters.num_samples += len(targets)
            dist, toks = compute_edit_distance(criterion.decode(outputs),
                                               targets)
            meters.edit_distance += dist
            meters.num_tokens += toks
            timers.stop("metrics").start("ds_fetch")
        timers.stop("ds_fetch").stop("train_total")
        epoch_time = time.time() - start_time
        if args.world_size > 1:
            meters.sync()
        if world_rank == 0:
            logging.info(
                "Epoch {} complete. "
                "nUpdates {}, Loss {:.3f}, CER {:.3f}, Time {:.3f} (s)".format(
                    epoch + 1, num_updates, meters.avg_loss, meters.cer,
                    epoch_time), )
            logging.info("Evaluating validation set..")
        timers.start("test_total")
        val_loss, val_cer = test(model, criterion, val_loader, world_rank,
                                 args.world_size)
        timers.stop("test_total")
        if world_rank == 0:
            checkpoint(model, criterion, args.checkpoint_path,
                       (val_cer < min_val_cer))

            min_val_loss = min(val_loss, min_val_loss)
            min_val_cer = min(val_cer, min_val_cer)
            logging.info(
                "Validation Set: Loss {:.3f}, CER {:.3f}, "
                "Best Loss {:.3f}, Best CER {:.3f}".format(
                    val_loss, val_cer, min_val_loss, min_val_cer), )
            logging.info("Timing Info: " + ", ".join([
                "{} : {:.2f}ms".format(k, v * 1000.0)
                for k, v in timers.value().items()
            ]))
        scheduler.step()
        start_time = time.time()

    if is_distributed_train:
        torch.distributed.destroy_process_group()


def main():
    args = parse_args()
    if args.world_size > 1:
        torch.multiprocessing.spawn(train,
                                    args=(args, ),
                                    nprocs=args.world_size,
                                    join=True)
    else:
        train(0, args)


if __name__ == "__main__":
    main()
