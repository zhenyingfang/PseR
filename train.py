import torch
from torch.utils import data
import random

def S_train(step, args, net, loader_iter, optimizer, logger):
    net.train()
    total_loss = {}
    total_cost = []
    optimizer.zero_grad()

    for batch in range(args.batch_size):
        sample = next(loader_iter)
        data, vid_label, point_label = sample['data'], sample['vid_label'], sample['point_label']
        data = data.to(args.device)
        vid_label = vid_label.to(args.device)
        point_label = point_label.to(args.device)

        outputs = net(data, vid_label)
        cost, loss_dict = net.criterion(args, outputs, vid_label, point_label)

        total_cost.append(cost)
        if not torch.isnan(cost):
            for key in loss_dict.keys():
                if not (key in total_loss.keys()):
                    total_loss[key] = []
                if loss_dict[key] > 0:
                    total_loss[key] += [loss_dict[key].detach().cpu().item()]
                else:
                    total_loss[key] += [loss_dict[key]]
                    
    total_cost = sum(total_cost) / args.batch_size
    total_cost.backward()
    optimizer.step()

    for key in total_loss.keys():
        logger.log_value("loss/" + key, sum(total_loss[key]) / args.batch_size, step)

    return total_cost.detach().cpu().item()


def PseR_train(step, args, net, loader_iter, optimizer, logger):
    net.train()
    total_loss = {}
    total_cost = []
    optimizer.zero_grad()

    for batch in range(args.batch_size):
        sample = next(loader_iter)
        data, vid_label, point_label = sample['data'], sample['vid_label'], sample['point_label']
        data = data.to(args.device)
        vid_label = vid_label.to(args.device)
        point_label = point_label.to(args.device)
        seed_seg = sample["seed_seg"]
        seed_label = sample["seed_label"]
        seed_seg = seed_seg.to(args.device)
        seed_label = seed_label.to(args.device)

        outputs = net.forward_pser(
            data,
            vid_label,
            seed_seg,
            seed_label,
            sample
        )
        cost = outputs['rk_mil_loss'] + outputs['refine_iou_loss'] + outputs['neg_loss']
        loss_dict = {
            'rk_mil_loss': outputs['rk_mil_loss'],
            'refine_iou_loss': outputs['refine_iou_loss'],
            'neg_loss': outputs['neg_loss']
        }

        total_cost.append(cost)
        if not torch.isnan(cost):
            for key in loss_dict.keys():
                if not (key in total_loss.keys()):
                    total_loss[key] = []
                if loss_dict[key] > 0:
                    total_loss[key] += [loss_dict[key].detach().cpu().item()]
                else:
                    total_loss[key] += [loss_dict[key]]
                    
    total_cost = sum(total_cost) / args.batch_size
    total_cost.backward()
    optimizer.step()

    for key in total_loss.keys():
        logger.log_value("loss/" + key, sum(total_loss[key]) / args.batch_size, step)

    return total_cost.detach().cpu().item()
