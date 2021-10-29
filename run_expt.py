import os, csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args
from train import train
from variable_width_resnet import resnet50vw, resnet18vw, resnet10vw

def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    # Data Modification
    parser.add_argument('--resample', default=False, action='store_true') # only works with confounder
    parser.add_argument('--swap_val_and_train', default=False, action='store_true') # only works with confounder
    parser.add_argument('--combine_val_test', default=False, action='store_true') # only works with confounder, train=True (currently always true), DOES NOT WORK IF SUBSAMPLE GROUP (MOVE TO -1)
    # Cross validation
    parser.add_argument('--cross_validate', default=False, action='store_true') # only works with combine_val_test
    parser.add_argument('--cross_validate_total_splits', type=int, default=2) # used with cross validate
    parser.add_argument('--cross_validate_split_num', type=int, default=0) # used with cross validate -- zero indexed. split_num = 0 ... cross_validate_total_splits-1
    # Resume?
    parser.add_argument('--resume', default=False, action='store_true')
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--subsample_to_minority', action='store_true', default=False)
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    # Objective
    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    parser.add_argument('--hinge', default=False, action='store_true')

    # Model
    parser.add_argument(
        '--model',
        choices=model_attributes.keys(),
        default='resnet50')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)
    parser.add_argument('--resnet_width', type=int, default=None)

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=4, help='number of (additional) epochs to run')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--step_scheduler', action='store_true', default=False)
    parser.add_argument('--scheduler_step', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)

    # move some worst group train data over to test
    parser.add_argument('--worst_group_train_to_test', action='store_true', default=False)
    parser.add_argument('--percent_to_move', type=float, default=0.5)
    parser.add_argument('--move_to_set', type=int, default=-1) # 0=train, 1=val, 2=test

    # 
    parser.add_argument('--reduce_group_size', action='store_true', default=False)
    parser.add_argument('--reduce_all_size', action='store_true', default=False) # uses percent_reduce, reduce_group_size_setidx, move_to_set. only for celebA
    parser.add_argument('--reduce_group_size_groupidx', type=int, default=0)
    parser.add_argument('--reduce_group_size_setidx', type=int, default=0)
    parser.add_argument('--percent_reduce', type=float, default=0.5)

    args = parser.parse_args()
    check_args(args)

    # BERT-specific configs copied over from run_glue.py
    if args.model == 'bert':
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    if os.path.exists(args.log_dir) and args.resume:
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    log_args(args, logger)

    set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        if args.combine_val_test:
            train_data, val_data = prepare_data(args, train=True)
        else:
            train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)

    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':4, 'pin_memory':True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)

    # =====================
    ## Initialize model start
    # =====================
    pretrained = not args.train_from_scratch
    if resume:
        pth_files_full_names = os.listdir(args.log_dir)
        if "last_model.pth" in pth_files_full_names:
            model = torch.load(os.path.join(args.log_dir, 'last_model.pth'))
        else:
            pth_files = sorted([int(f.split('_')[0]) for f in pth_files_full_names if f.split('_')[0].isnumeric()])
            last_pth = pth_files[-1]
            
            for filename in ['train', 'val', 'test']:
                filename_csv =  f'{filename}.csv'
                file_pth = os.path.join(args.log_dir, filename_csv)
                
                df = pd.read_csv(file_pth)
                df = df[:last_pth+1]
                df.to_csv(file_pth, index=False)
            
            model = torch.load(os.path.join(args.log_dir, f'{last_pth}_model.pth'))
        d = train_data.input_size()[0]
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False

    # resnet imagenet pretrain
    elif args.model == 'resnet152':
        model = torchvision.models.resnet152(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)

    # vgg
    elif args.model == 'vgg11_bn':
        model = torchvision.models.vgg11_bn(pretrained=pretrained)
        d = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = nn.Linear(d, n_classes)
    elif args.model =='vgg13_bn':
        model = torchvision.models.vgg13_bn(pretrained=pretrained)
        d = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = nn.Linear(d, n_classes)
    elif args.model =='vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=pretrained)
        d = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = nn.Linear(d, n_classes)
    elif args.model =='vgg19_bn':
        model = torchvision.models.vgg19_bn(pretrained=pretrained)
        d = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = nn.Linear(d, n_classes)

    # misc
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model =='resnet50vw':
        assert not pretrained
        assert args.resnet_width is not None
        model = resnet50vw(args.resnet_width, num_classes=n_classes)
    elif args.model =='resnet18vw':
        assert not pretrained
        assert args.resnet_width is not None
        model = resnet18vw(args.resnet_width, num_classes=n_classes)
    elif args.model =='resnet10vw':
        assert not pretrained
        assert args.resnet_width is not None
        model = resnet10vw(args.resnet_width, num_classes=n_classes)
    
    # misc resnet50 pretrain
    elif args.model == 'barlowtwins_resnet50':
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'weaksup_resnet50':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'facenet':
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained="vggface2", num_classes=n_classes, classify=True)
    elif args.model == 'resnext101_32x8d_wsl':
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
        
    # text models
    elif args.model == 'bert':
        assert args.dataset == 'MultiNLI'

        from pytorch_transformers import BertConfig, BertForSequenceClassification
        config_class = BertConfig
        model_class = BertForSequenceClassification

        config = config_class.from_pretrained(
            'bert-base-uncased',
            num_labels=3,
            finetuning_task='mnli')
        model = model_class.from_pretrained(
            'bert-base-uncased',
            from_tf=False,
            config=config)
    else:
        raise ValueError('Model not recognized.')

    # =====================
    ## Initialize model end
    # =====================

    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset in ['CelebA', 'CUB'] # Only supports binary
        def hinge_loss(yhat, y):
            # The torch loss takes in three arguments so we need to split yhat
            # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
            # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
            # so we need to swap yhat[:, 0] and yhat[:, 1]...
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)
        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, 'test.csv'))
        epoch_offset = df.loc[len(df)-1,'epoch']+1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset=0
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode)

    train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args, epoch_offset=epoch_offset)

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio
    
    if args.step_scheduler and args.scheduler:
        raise Exception("please only set flag of 1 lr scheduler")

    if args.worst_group_train_to_test and args.reduce_third_size_group:
        raise Exception("please only set flag of 1 lr scheduler")



if __name__=='__main__':
    main()
