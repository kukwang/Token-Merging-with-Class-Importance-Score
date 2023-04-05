import os
import torch


from sampler import RASampler

def build_dataloader(args, train_set, val_set, num_tasks, global_rank):
    if args.repeated_aug:
        train_sampler = RASampler(
            train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        train_sampler = torch.utils.data.DistributedSampler(
            train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if args.dist_eval:
        if len(val_set) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        val_sampler = torch.utils.data.DistributedSampler(
            val_set, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_set)


    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if val_set is not None:
        val_loader = torch.utils.data.DataLoader(
            val_set, sampler=val_sampler,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        val_loader = None
        
    return train_loader, val_loader