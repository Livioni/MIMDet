#!/usr/bin/env python
"""Training script using the new "LazyConfig" python config files.
This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.
Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging,cv2
import torch
import detectron2.data.transforms as T

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret
    
output_pool = []    
def hook(module, input, output):
    output_pool.append(output.shape)

def do_inference(args, model):
    image = cv2.imread(args.input)
    image = T.ResizeShortestEdge(short_edge_length=800, max_size=1333).get_transform(image).apply_image(image)
    img_tensor = torch.as_tensor(image.astype("float32").transpose(2,0,1))
    model_input = [{'image': img_tensor}]
    model.eval()
    input_pool = []

    layers_to_hook = [model.backbone.fpn_output2,
                      model.backbone.fpn_output3,
                      model.backbone.fpn_output4,
                      model.backbone.fpn_output5]  # 用实际的层名替换
    handles = []
    for layer in layers_to_hook:
        handle = layer.register_forward_hook(hook)
        handles.append(handle)

    with torch.no_grad():
        outputs = model(model_input)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("coco_val_2017"), scale=1.2)
    for output in outputs:
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        result_img = v.get_image()[:, :, ::-1]
        #print inference result
        # print("Instance Class:",output["instances"].pred_classes)
        # print("Instance Bboxes:",output["instances"].pred_boxes)
        # print("Instance Confidence:",output["instances"].scores)
        # 显示可视化结果
        # cv2.imshow("Visualization", result_img)
        image_name = args.input.split("/")[-1]
        cv2.imwrite(args.output + image_name,result_img)
    print("output_pool:",output_pool)
    for handle in handles:
        handle.remove()
    return

def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(
        model, train_loader, optim
    )
    checkpointer = DetectionCheckpointer(model, cfg.train.output_dir, trainer=trainer,)
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    elif args.inference:
        model = instantiate(cfg.model)  # returns a torch.nn.Module
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        do_inference(args, model)
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", default="")
    parser.add_argument("--master_port", default="")
    parser.add_argument("--inference", action="store_true", help="perform inference only")
    parser.add_argument(
        "--input",
        type=str,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
    )
    args = parser.parse_args()
    args.machine_rank = args.node_rank
    if args.master_addr or args.master_port:
        assert args.master_addr and args.master_port
        args.dist_url = f"tcp://{args.master_addr}:{args.master_port}"
    print(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
