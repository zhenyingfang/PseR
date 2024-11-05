import os
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
from tensorboard_logger import Logger
import time
import datetime
import json
from torch.optim.lr_scheduler import StepLR

import options
import utils.model_utils as utils
from dataset import dataset
from models.model_base import S_Model
from train import S_train, PseR_train
from test import S_test, PseR_test
from log import save_config, initial_log, save_best_record

class_name_lst = [
        'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
        'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
        'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump', 'JavelinThrow',
        'LongJump', 'PoleVault', 'Shotput', 'SoccerPenalty', 'TennisSwing',
        'ThrowDiscus', 'VolleyballSpiking'
    ]

def main(args):
    # >> Initialize the task
    save_config(args, os.path.join(args.output_path_s1, "config.json"))
    utils.set_seed(args.seed)
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    # --------------------------------------------------Snippet-level Optimization-------------------------------------------------------#
    if args.stage == 1:
        model = S_Model(args)
        model = model.to(args.device)
        train_loader = data.DataLoader(dataset(args, phase="train", sample="random", stage=args.stage), 
                                       batch_size=1, shuffle=True, num_workers=args.num_workers)
        test_loader = data.DataLoader(dataset(args, phase="test", sample="random", stage=args.stage),
                                    batch_size=1, shuffle=False, num_workers=args.num_workers)
        test_info = args.test_info
        if args.mode == 'train':
            logger = Logger(args.log_path_s1)
            log_filepath = os.path.join(args.log_path_s1, '{}.score'.format(args.dataset))
            initial_log(log_filepath, args)

            model.memory.init(args, model, train_loader)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
            
            best_mAP = -1
            start_time = time.time()
            process_bar = tqdm(range(1, args.num_iters + 1), total=args.num_iters)
            for step in process_bar:
                process_bar.set_description('PseR-Model')
                if (step - 1) % (len(train_loader) // args.batch_size) == 0:
                    loader_iter = iter(train_loader)
                loss = S_train(step, args, model, loader_iter, optimizer, logger)
                
                if step % args.test_iter == 0:
                    test_mAP = S_test(model, args, test_loader, logger, step, test_info)
                    test_info['loss'].append(loss)
                    test_info["elapsed"].append(str(datetime.timedelta(seconds = time.time() - start_time)))
                    test_info["now"].append(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    if test_mAP > best_mAP:
                        best_mAP = test_mAP
                        save_best_record(test_info, log_filepath)
                        torch.save(model.state_dict(), os.path.join(args.model_path_s1, "model1_seed_{}.pkl".format(args.seed)))
                        
                    print("\n Current test_mAP:{:.4f} best_mAP:{:.4f}".format(test_mAP, best_mAP))
                    logger.log_value('acc/best mAP', best_mAP, step)

            # >> PseR train
            start_time = time.time()
            process_bar = tqdm(range(1, args.pser_num_iters + 1), total=args.pser_num_iters)
            for step in process_bar:
                if step > args.pser_end_iter:
                    break
                process_bar.set_description('PseR-Model')
                if (step - 1) % (len(train_loader) // args.batch_size) == 0:
                    loader_iter = iter(train_loader)
                loss = PseR_train(step, args, model, loader_iter, optimizer, logger)
                
                if step % args.pser_test_iter == 0:
                    rk_mil_losss = PseR_test(model, args, train_loader)
                    rk_mil_losss = np.array(rk_mil_losss).mean()
                    print("Current rk loss: {:.4f}".format(rk_mil_losss))
        else:
            model.load_state_dict(torch.load(os.path.join(args.model_path_s1, "model1_seed_{}.pkl".format(args.seed))))
            S_test(model, args, test_loader, None, 0, test_info, subset='test')
            S_test(model, args, train_loader, None, 0, test_info, subset='train')
    else:
        raise f"not support now stage {args.stage}"


if __name__ == "__main__":
    args = options.parse_args()
    main(args)
