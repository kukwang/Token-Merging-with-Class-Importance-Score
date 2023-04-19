from utils import str2bool
import argparse

def add_arguments(parser:argparse.ArgumentParser):
    dataset_list = ['CIFAR10', 'CIFAR100', 'ImageNet']

    # Device setting
    parser.add_argument('--device', default='cuda',
                        help='device (default: cuda)')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed (default: 10)')
    parser.add_argument('--dist_eval', type=str2bool, default=False,
                    help='Enabling distributed evaluation')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of distributed processes (default: 1)')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training (default: env://)')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU (default: True)')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log (default: None)')


    # Datasets parameters
    parser.add_argument('--data_path', default='', metavar='DIR',
                        help='path to dataset (default: "")')
    parser.add_argument('--dataset_name', type=str, default="ImageNet", choices=dataset_list,
                        help='name of the dataset (default: ImageNet)')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    
    
    # Augmentation parameters
    parser.add_argument('--resize', type=int, default=256,
                        help='resize size (default: 256)')
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='training interpolation (random, bilinear, bicubic) (default: "bicubic")')

    parser.add_argument('--repeated_aug', action='store_true')
    parser.set_defaults(repeated_aug=True)


    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='random erase count (default: 1)')
    
    
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0 (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0 (default 1.0)')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='probability of performing mixup or cutmix when either/both is enabled (default 1.0)')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='probability of switching to cutmix when both mixup and cutmix enabled (default: 0.5)')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='how to apply mixup/cutmix params. Per "batch", "pair", or "elem" (default: batch)')


    # Model parameters
    parser.add_argument('--model_name', type=str, default='deit_small_patch16_224', metavar='MODEL',
                        help='name of model to train (default: deit)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='image input size (default: 224)')
    parser.add_argument('--mlp_dim', type=int, default=2048,
                        help='dimension of mlp layer (default: 2048)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='drop path rate (default: 0.1)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='drop out ratio (default: 0.0)')
    parser.add_argument('--emb_dropout', type=float, default=0.1,
                        help='drop out ratio in embedding (default: 0.1)')
    parser.add_argument('--model_ema', action='store_true',
                        help='enable ema')
    parser.add_argument('--head_dim', type=int,
                        help='dimension of the head')
    parser.add_argument('--save_path', default=None,
                        help='save path (default: None)')
    parser.add_argument('--pt_dl', type=str, default=None,
                        help='download pretrained model (default: None)')
    parser.add_argument('--pt_local', default=None,
                        help='pretrained model path (default: None)')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving (default:"")')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint (default:"")')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)


    # Optimization parameters
    parser.add_argument('--opt', type=str, default='adamw', metavar='OPTIMIZER',
                        help='optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', type=float, default=1e-8, metavar='EPSILON',
                        help='optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', type=float, default=None, nargs='+', metavar='BETA',
                        help='optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None,
                        help="""Final value of the weight decay.
                        We use a cosine schedule for WD and using a larger decay by
                        the end of training improves performance for ViTs. (default: None)""")


    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    # parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0 (default: -1)')
    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
    #                     help='patience epochs for Plateau LR scheduler (default: 10')
    # parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
    #                     help='LR decay rate (default: 0.1)')


    # Training/Evaluation parameters
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint (default: '')')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation mode (default: False)')
    parser.add_argument('--use_amp', default=False,
                        help='use "torch.cuda.amp.autocast" at evaluation (default: False)')
    parser.add_argument('--train_ratio', type=float, default=0.0,
                        help='train/val split ratio (default: 0.0, not split)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch (default: 0)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='training epoch (default: 300)')
    parser.add_argument('--update_freq', type=int, default=1,
                        help='gradient accumulation steps (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size at evaluation (default: 128)')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--training_debug', type=int, default=0,
                        help='training iters at one epoch while debugging (defualt: 10)')


    # Distillation parameters
    parser.add_argument('--teacher_model', default='regnety_160', type=str, metavar='MODEL',
                        help='name of teacher model to train (default: regnety_160)')
    parser.add_argument('--teacher_path', type=str, default='',
                        help='path of teacher model (default: "")')
    parser.add_argument('--distillation_type', default='none', choices=['none', 'soft', 'hard'], type=str,
                        help='distillation type [none, soft, hard] (default: none)')
    parser.add_argument('--distillation_alpha', default=0.5, type=float,
                        help='weight parameter between normal loss and distillation loss (default: 0.5)')
    parser.add_argument('--distillation_tau', default=1.0, type=float,
                        help='temperature for distillation (default: 1.0)')

    # personal params
    parser.add_argument('--tome_r', type=int, default=0,
                        help='reduce token number in ToMe (default: 0)')
    parser.add_argument('--mymodel', action='store_true',
                        help='use my model')

    parser.add_argument('--drop_loc', default='(3, 6, 9)', type=str, 
                        help='the layer indices for shrinking inattentive tokens (default: (3, 6, 9))')
    parser.add_argument('--keep_rate', type=float, default=1.0,
                        help='token split rate in merging layer (default: 1.0)')
    parser.add_argument('--trade_off', type=float, default=1.0,
                        help='trade of Evo-ViT (default: 0.5)')

    parser.add_argument('--custom', action='store_true',
                        help='use custom code')
    

    return parser