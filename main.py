import torch
import numpy as np
import options
from datasets import RGDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import model, loss
import os
from torch import nn

import train
from test import test_epoch


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - 30], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler

def build_model(args):
    if args.submodel_name == "cofinal" or args.submodel_name == "cofinal_1":
        from models.model import cofinal as Model
        print("cofinal...")
        model = Model(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder,
                       args.n_decoder, args.n_query, args.dropout, args.activate_regular_restrictions)
    
    if args.submodel_name == "cofinal_2":
        from models.model import cofinal_2 as Model
        print("cofinal_2...")
        model = Model(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder,
                       args.n_decoder, args.n_query, args.dropout, args.activate_regular_restrictions)

    return model

if __name__ == '__main__':
    from logger import Logger
    
    args = options.parser.parse_args()
    setup_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device: ', device)   
    keep = False
    if args.ab != 0:
        keep = True
        args.submodel_name = args.submodel_name + "_" + str(args.ab)
    
    if args.model_name in ['Ball', 'Clubs', 'Hoop' , 'Ribbon']:
        dataset = 'RG'
    else:
        dataset = 'fis-v'
        
    dataset += '/I3D' if args.in_dim == 2048 else '/VST'
        
    save_path = f"./temp_log/{dataset}/"
    if not os.path.exists(save_path + args.submodel_name):
        os.makedirs(save_path + args.submodel_name)

    name = args.submodel_name
    if 'cofinal' in name:
        use_etf = True
    else:
        use_etf = False

    phase = 'test' if args.test else 'train'
    logger = Logger( 
        file_name = save_path + f"/{args.submodel_name}/{phase}_log-{name}-{args.action_type}.txt", 
        file_mode = "w+", 
        should_flush = True
    )
    '''
    1. load data
    '''
    '''
    train data
    '''
    train_data = RGDataset(args.video_path, args.train_label_path, clip_num=args.clip_num,
                           action_type=args.action_type)
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=8)

    '''
    test data
    '''
    test_data = RGDataset(args.video_path, args.test_label_path, clip_num=args.clip_num,
                          action_type=args.action_type, train=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    print('=============Load dataset successfully=============')

    '''
    2. load model
    '''
    model = build_model(args).to(device)
    loss_fn = loss.LossFun(args.alpha, args.margin, True, args.loss_align)
    train_fn = train.train_epoch
    if args.ckpt is not None:
        path = args.ckpt
        print( "load param:", path)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
    print('=============Load model successfully=============')

    print(args)

    '''
    test mode
    '''
    if args.test:
        test_loss, coef, rl2, preds, labels, feat = \
            test_epoch(0, model, test_loader, None, device, args, use_etf, True, True, keep=keep)
        print('Test Loss: {:.4f}\tTest Coef: {:.3f}\tR-L2: {:.3f}'.format(test_loss, coef, rl2))
        
        np.savez(save_path + f"{args.submodel_name}/{args.action_type}_{args.submodel_name}", 
            pred=preds, gt=labels, q1=feat[0], other=feat[1], in_feat=feat[2]
        )
        
        raise SystemExit

    '''
    3. record
    '''
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)
    logger = SummaryWriter(os.path.join('./logs/', args.model_name))
    best_coef, best_rl2, best_epoch = -1, -1, -1
    final_train_loss, final_train_coef, final_test_loss, final_test_coef = 0, 0, 0, 0

    '''
    4. train
    '''
    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)
    print('=============Begin training=============')
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None

    for epc in range(args.epoch):
        if args.warmup and epc < args.warmup:
            warmup.step()
        
        avg_loss, train_coef = train_fn(epc, model, loss_fn, train_loader, optim, logger, device, args, use_etf)
        if scheduler is not None and (args.lr_decay != 'cos' or epc >= args.warmup):
            scheduler.step()
        test_loss, test_coef, test_rl2, preds, labels, feat = test_epoch(epc, model, test_loader, logger, device, args, use_etf, True, True, keep = keep)
        if test_coef > best_coef:
            best_coef, best_rl2, best_epoch = test_coef, test_rl2, epc
            torch.save(model.state_dict(), save_path + f"{args.submodel_name}/" + args.model_name + '_best.pkl')
            torch.save(model.state_dict(),  "./ckpt/" + args.model_name + '_best.pkl')

            np.savez(save_path + f"/{args.submodel_name}/{args.model_name}_{args.action_type}", preds=preds.squeeze(), 
                labels=labels.squeeze(), feat=feat)

        print('Epoch: {}\tLoss: {:.4f}\tTrain Coef: {:.3f}\tTest Loss: {:.4f}\tTest Coef: {:.6f}\tTest rl2: {:.4f}'
              .format(epc, avg_loss, train_coef, test_loss, test_coef, test_rl2))
        if epc == args.epoch - 1:
            final_train_loss, final_train_coef, final_test_loss, final_test_coef = \
                avg_loss, train_coef, test_loss, test_coef
    
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
    print('Best Test Eopch: {}\t'
          'Best Test Coef: {:.6f}\t'
          'Best Test R-L2: {:.3f}'.format(best_epoch, best_coef, best_rl2))