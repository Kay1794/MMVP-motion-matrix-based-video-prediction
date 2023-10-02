import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse
import os
from shutil import copytree, copy
from utils import *

# from tmp_mode import *
from model import *
from module import *
import configs
from copy import deepcopy as cp
import lpips
import gc
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main(config,args):
    num_epochs = args.nepoch
    need_log = args.log
    num_workers = args.nworker

    start_epoch = 0
    best_psnr = 0.

    # config log info for training mode
    torch.autograd.set_detect_anomaly(True)
    if args.mode == 'train' and need_log:
        logger_root = args.logpath if args.logpath != '' else 'results'

        if args.resume == '' or args.weight_only:
            time_stamp = time.strftime("%m-%d_%H-%M")

            model_save_path = check_folder(os.path.join(logger_root, args.dataset))
            model_save_path = check_folder(os.path.join(model_save_path, args.method))
            model_save_path = check_folder(os.path.join(model_save_path, args.exp_name
                                                        + '_'+time_stamp))
            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

            # Copy the code files as logs
            copytree('data', os.path.join(model_save_path, 'data'))
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            for f in python_files:
                copy(f, model_save_path)
        else:
#
            model_save_path = args.resume[:args.resume.rfind('/')]
            
            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            
            saver.flush()
            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")
            saver.write('Running log: '+str(config)+'\n')
        
            saver.flush()
    else:
        model_save_path = None
        saver = open('tmp.txt', "a")

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    # Load data

    if args.mode == 'train':
        trainset = PredDataset(config=config, split='test' if args.overfit else 'train', is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True,num_workers=num_workers)
        config['train_disp_freq'] = max(1,len(trainset) // args.batch // args.display_freq)
        config['train_steps'] = len(trainloader)
        valset = PredDataset(config=config, split='test', is_train=False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False, num_workers=num_workers)
        config['val_disp_freq'] = max(1,len(valset) // args.batch // args.display_freq)
        print("Training dataset size:", len(trainset))
        print("Validation dataset size:", len(valset))
        if args.log:
            config['model_save_path'] = model_save_path

    elif args.mode in ['val','test']:

        valset = PredDataset(config=config, split='test', is_train=False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False, num_workers=num_workers)
        config['val_disp_freq'] = len(valset) // args.batch // args.display_freq
        model_save_path = args.resume[:args.resume.rfind('/')]
        log_file_name = os.path.join(model_save_path, 'validation_log.txt')
        saver = open(log_file_name, "a")
        saver.write("Validation on : {}\n".format(str(args.resume)))
        saver.flush()
        print("Validation dataset size:", len(valset))


    # ----------------------------#
    # build model
    if config['method'] == 'ours' or config['method'] == 'ours_e2e':
        model = Model(config)
    elif config['method'] == 'simvp':
        if config['dataset'] == 'caltech':
            model = SimVP(config=config,hid_S=64, hid_T=128,N_S=1, N_T=3)
        else:
            model = SimVP(config=config)
    elif config['method'] == 'deform':
        model = DeformModel(config=config)
    elif config['method'] == 'ours_warp':
        model = WarpModel(config=config)
    
    

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model Parameter num: ',pytorch_total_params)
    if args.log:
        saver.write('Model Parameter num: '+str(pytorch_total_params)+'\n')
        saver.flush()
    
    model = model.to(device)
    model = nn.DataParallel(model)
    # ----------------------------#

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.0005)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable Parameter num: ',pytorch_total_params)
    if args.log:
        saver.write('Trainable Parameter num: '+str(pytorch_total_params)+'\n')
        saver.flush()

    # specify creterion
    reduction = 'mean'
    motion_reduction = 'sum'
    criterion = {'recon': torch.nn.MSELoss(reduction=reduction)}
    module = Module(model, config, optimizer, criterion)
    module.loss_list = args.loss_list
    # ------------------------------#
    # load model
    if args.resume != '' or args.mode in ['val', 'test']:
        checkpoint = torch.load(args.resume)

        message = module.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        # message = module.model.load_state_dict(checkpoint,strict=False) # only for simvp loading
        print(message)
        if not args.weight_only:
            start_epoch = checkpoint['epoch'] + 1
            if 'best_psnr' in checkpoint:
                
                best_psnr = checkpoint['best_psnr']
                if np.isinf(best_psnr):
                    best_psnr = -1
            if not module.scheduler is None:
                module.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

            module.epoch = start_epoch -1

    # ------------------------------#
    if args.mode == 'train':
        for epoch in range(start_epoch, num_epochs + 1):
            selected_display = []
            selected_display_matrix = []
            start_time = time.time()
            lr = module.optimizer.param_groups[0]['lr']
            print("Epoch {}, learning rate {}".format(epoch, lr))

            if need_log:
                saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
                saver.flush()

            metrics = {}
            metrics['total'] = AverageMeter('Total loss', ':.6f')  # for motion prediction error
            for key in args.loss_list:
                metrics[key] = AverageMeter(key + ' loss', ':.6f')

            module.model.train()
            module.epoch = epoch-1

            it = time.time()


            for i, sample in enumerate(trainloader, 0):
                # if i != 3015:
                #     continue

                img,gt = sample

                data = {}
                data['input_img'] = img.type(dtype=torch.float32)
                data['gt_img'] = gt.type(dtype=torch.float32)

                

                output_list, loss_dict = module.step(data,epoch)
                recon_img = output_list['recon_img']
                
                if 'pred_sim_matrix' in output_list:
                    gt_sim_matrix = output_list['gt_sim_matrix'][0][0][-1]
                    pred_sim_matrix = output_list['pred_sim_matrix'][0][0][0] # (h*w, h*w)
                
                metrics = update_metrics(metrics,loss_dict)
                recon_img = recon_img.reshape(img.shape[0], config['fut_len'], config['in_res'][0],
                                                config['in_res'][1], -1)
                if i % config['train_disp_freq'] == 0 or i == 3015:
                    selected_display.append(
                        cp((data['input_img'][0].cpu().detach().numpy().copy(), recon_img[0].cpu().detach().numpy().copy())))
                    if 'pred_sim_matrix' in output_list and (config['fut_len'] == 1):
                        selected_display_matrix.append(cp((gt_sim_matrix,pred_sim_matrix)))
                    message = metric_print(metrics,epoch,i,str(time.time() - it))
                    print(message)
                    it = time.time()
                del output_list
                gc.collect()
            
                
                
                # if i > 0:
                #     break
            

            if not module.scheduler is None:
                module.scheduler.step()
            message = metric_print(metrics, epoch, -1, str(time.time() - start_time),True)
            print(message)

            

            if need_log:
                saver.write(message+'\n')
                saver.flush()
                if args.energy_save_mode and ((epoch % int(config['t_period'][0])) < (0.8 * (config['t_period'][0]))) and (epoch % 5 !=0):
                    print('---------------------------')
                    continue
                print('Validation on Epoch ',epoch)
                print('---------------------------')
                save_dict = {'epoch': epoch,
                                'model_state_dict': module.model.state_dict(),
                                'optimizer_state_dict': module.optimizer.state_dict(),
                                'loss': metrics['total'].avg,
                                'best_psnr': best_psnr}
                #if epoch > 15:
                val_metrics = validate(valloader, module, config, model_save_path, saver, epoch)
                
                
                save_dict = {'epoch': epoch,
                            'model_state_dict': module.model.state_dict(),
                            'optimizer_state_dict': module.optimizer.state_dict(),
                            'loss': metrics['total'].avg,
                            'best_psnr': best_psnr}
                if not module.scheduler is None:
                    save_dict['scheduler_state_dict'] = module.scheduler.state_dict()
                
                if val_metrics['psnr'].avg > best_psnr:
                    best_psnr = val_metrics['psnr'].avg
                    save_dict['best_psnr'] = best_psnr
                    torch.save(save_dict, os.path.join(model_save_path, 'best_psnr_'+str(epoch // 10)+'to'+str(epoch // 10+1)+'.pth'))
                if (epoch > 0) and (epoch %10 == 0):
                    best_psnr = 0. #reinitialize
                torch.save(save_dict, os.path.join(model_save_path, 'latest.pth'))
                if config['nepoch'] < 300:
                    vis_flag = True
                elif (epoch % 50 ==0) or (epoch > config['nepoch']-50):
                        vis_flag = True
                else:
                    vis_flag = False 
                if vis_flag:
                    if not args.overfit:
                        visualization_check_video(model_save_path, epoch, selected_display, valid=(config['range'] == 255),is_train=True,config=config,long_term = config['fut_len'] > 1)
                        if config['method'].find('ours') > -1 and len(selected_display_matrix) > 0:
                            visualization_check_video(model_save_path, epoch, selected_display_matrix, valid=(config['range'] == 255),is_train=True,matrix=True,config=config)

            del selected_display
            del selected_display_matrix
                
            if not need_log:
                validate(valloader, module, config, None, None, None)
            gc.collect()
            print('---------------------------')
 

    elif args.mode == 'val':
        print('Validate on epoch ', module.epoch)
        validate(valloader, module, config, model_save_path, saver, -1)
    elif args.mode == 'test':
        print('Test on epoch ', module.epoch)
        test(valloader, module, config, model_save_path, saver,-1)


def validate(valloader,module,config,model_save_path,saver,epoch):
    
    
    module.model.eval()
    val_metrics = {}
    eval_metrics = {}
    for key in config['loss_list']:
        val_metrics[key] = AverageMeter(key + ' loss', ':.6f',is_val=True)
    for key in config['eval_list']:
        eval_metrics[key] = AverageMeter(key, ':.6f',is_val=True)
    val_metrics['total'] = AverageMeter('Total loss', ':.6f',is_val=True)  # for motion prediction error
    it = time.time()
    start_time = time.time()
    selected_display_matrix = []
    selected_display = []
    check_id = 2127
    with torch.no_grad():

        for i, sample in enumerate(valloader, 0):
            # if not((i % config['val_disp_freq'] == 0) and (i > 0)):
            #         continue
            # if i != 3015:
            #     continue
            # if i != check_id:
            #     continue
            img,gt = sample
            data = {}
            data['input_img'] = img.type(dtype=torch.float32)
            data['gt_img'] = gt.type(dtype=torch.float32)

            output_list, loss_dict = module.val(data,epoch)
            recon_img = output_list['recon_img']
            data['gt_img'] = data['gt_img']
            if 'pred_sim_matrix' in output_list:
                gt_sim_matrix = output_list['gt_sim_matrix'][0][0][-1]
                
                pred_sim_matrix = output_list['pred_sim_matrix'][0][0][0] # (h*w, h*w)

            val_metrics = update_metrics(val_metrics,loss_dict)
            recon_img = recon_img.reshape(img.shape[0],config['fut_len'],config['in_res'][0],config['in_res'][1],-1)
            

            if config['only_val_last']:
                eval_metrics = image_evaluation(recon_img[:,-1].cpu().detach().numpy(),data['gt_img'][:,-1].cpu().detach().numpy(),eval_metrics,valid=(config['range']==255))
            else:
                eval_metrics = image_evaluation(recon_img.cpu().detach().numpy(),data['gt_img'].cpu().detach().numpy(),eval_metrics,valid=(config['range']==255))
            
            if i % config['val_disp_freq'] == 0 or i == check_id:
                selected_display.append(cp((data['input_img'][0].cpu().detach().numpy(), recon_img[0].cpu().detach().numpy())))
                if 'pred_sim_matrix' in output_list and (config['fut_len'] == 1):
                    selected_display_matrix.append(cp((gt_sim_matrix,pred_sim_matrix)))
                message = metric_print(val_metrics, epoch, i, str(time.time() - it))
                print(message)
                it = time.time()
            del output_list
            #gc.collect()
            
            # if i > 0:
            #     break

    #np.save('metrics.npy',np.asarray(metrics_to_save))
    val_metrics = {**val_metrics,**eval_metrics}
    message = metric_print(val_metrics, epoch, -1, str(time.time() - start_time), True)
    print(message)

    if config['nepoch'] < 300 or epoch < 0:
        vis_flag = True
    elif (epoch % 50 ==0) or (epoch > config['nepoch']-50):
            vis_flag = True
    else:
        vis_flag = False 
    
    if vis_flag:
        visualization_check_video(model_save_path, epoch, selected_display, valid=(config['range'] == 255),config=config,long_term = config['fut_len'] > 1)
        if config['method'].find('ours') > -1 and len(selected_display_matrix) > 0 :
            visualization_check_video(model_save_path, epoch, selected_display_matrix, valid=(config['range'] == 255),matrix=True,config=config)

    if not saver is None:
        saver.write('Validation epoch ' + str(epoch) + '_' + message+'\n')
    del selected_display
    del selected_display_matrix
    #gc.collect()
    # hf.close()


    return val_metrics

def test(valloader,module,config,model_save_path,saver,epoch):
    print('Test on Epoch ',epoch)
    print('---------------------------')
    module.model.eval()
    val_metrics = {}
    eval_metrics = {}
    for key in config['loss_list']:
        val_metrics[key] = AverageMeter(key + ' loss', ':.6f',is_val=True)
    for key in config['eval_list']:
        eval_metrics[key] = AverageMeter(key, ':.6f',is_val=True)
    val_metrics['total'] = AverageMeter('Total loss', ':.6f',is_val=True)  # for motion prediction error
    it = time.time()
    start_time = time.time()
    selected_display_matrix = []
    selected_display = []
    with torch.no_grad():

        for i, sample in enumerate(valloader, 0):
            img,gt = sample
            data = {}
            data['input_img'] = img.type(dtype=torch.float32)[:,:config['prev_len']]
            data['gt_img'] = gt.type(dtype=torch.float32)

            output_list= module.test(data,epoch)
            recon_img = output_list['recon_img']

            recon_img = recon_img.detach().cpu().numpy().reshape(img.shape[0],config['fut_len'],config['in_res'][0],config['in_res'][1],-1) # B,T,H,W,C
            prev_img = data['input_img'].detach().cpu().numpy().reshape(img.shape[0],config['prev_len'],config['in_res'][0],config['in_res'][1],-1) # B,T,H,W,C
            gt_img = gt.detach().cpu().numpy().reshape(img.shape[0],config['fut_len'],config['in_res'][0],config['in_res'][1],-1) # B,T,H,W,C
            visualization_check_video_testmode(model_save_path+'/test_visualization_'+(config['val_subset'])+'/',[prev_img,gt_img,recon_img],valid=(config['range'] == 255),config=config,iter_id=i)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #model related 
    parser.add_argument('--method', default='ours_e2e', type=str, help='Which method to be test',choices= ['ours_e2e'])
    parser.add_argument('--pred_type', default='highonly', help='motion prediction method',choices= ['highonly','highinter'])
    parser.add_argument('--pred_method', default='conv3d', help='motion predictor',choices= ['conv3d'])
    parser.add_argument('--base_channel', default=32, type=int, help='input image range')
    parser.add_argument('--pred_base_channel', default=32, type=int, help='input image range')
    parser.add_argument('--shuffle_setting', action='store_true', help='Pixel shuffle')
    parser.add_argument('--filter_block', action='store_true', help='Conv projection before similarity matrix')
    parser.add_argument('--add_feat_diff', action='store_true', help='add img diff infor to matrix prediction')
    parser.add_argument('--res_img_scale', default=1, type=int, help='The downsample scale of the residual image')
    parser.add_argument('--use_fix_mat_size', action='store_true', help='If fix the mat size for every dataset')
    parser.add_argument('--bilinear', action='store_true', help='Upsample / Transpose in reconstruction')
    parser.add_argument('--res_cat_img', action='store_true', help='add residule image by conducting weighted sum')
    parser.add_argument('--long_term', action='store_true', help='do long term forward')
    parser.add_argument('--fut_len', default=-1, type=int, help='If -1, follow the initial config parameter, otherwise, update config')
    parser.add_argument('--prev_len', default=-1, type=int, help='If -1, follow the initial config parameter, otherwise, update config')
    parser.add_argument('--rrdb_encoder_num', default=2, type=int, help='Use RRDB block numbers in RRDBEncoder')
    parser.add_argument('--rrdb_decoder_num', default=2, type=int, help='Use RRDB block numbers in RRDBDecoder')
    parser.add_argument('--rrdb_enhance_num', default=2, type=int, help='Use RRDB block numbers in ImageEnhancer')
    parser.add_argument('--use_direct_predictor', action='store_true', help='Use direct predictor for any future length ')
    parser.add_argument('--downsample_scale', nargs="*", type=int,default=[], help='Motion scale, use the last two')
    parser.add_argument('--translate_factor', default=1, type=int, help='Downsample ratio of translate block feature')
    parser.add_argument('--scale_in_use', default='all', help='how many scales of features used for composition',choices= ['all','3','2'])
    parser.add_argument('--img_pred_res', action='store_true', help='Use SplitFaster3DConv ')
    
    #data related
    parser.add_argument('--dataset', default='ucf', help='choose dataset',choices= ['mnist','ucf_4to1','strpm','kth'])
    parser.add_argument('--img_range', default=1, type=int, help='input image range')
    parser.add_argument('--flip_aug', action='store_true', help='flip augmentation')
    parser.add_argument('--rot_aug', action='store_true', help='rotation augmentation')
    parser.add_argument('--overfit', action='store_true', help='overfitting setting')
    parser.add_argument('--val_subset', default='hard', type=str, help='Which subset to use',choices= ['hard','intermediate','easy','all'])
    parser.add_argument('--train_length', default=1e4, type=int, help='Mnist training length. default is 1e4')
    
    
    # training related
    parser.add_argument('--mode', default=None, help='Train/Val mode',choices=['train','val','test'])
    parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--batch', default=32, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=10, type=int, help='Number of epochs')
    parser.add_argument('--display_freq', default=6, type=int, help='display frequency')
    parser.add_argument('--nworker', default=0, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--step', default=-1, type=int, help='lr_scheduler decay step')
    parser.add_argument('--milestone', nargs="*", type=int,default=[], help='MultistepLR parameter')
    parser.add_argument('--decay', default=0.5, type=float, help='lr_scheduler decay rate')
    parser.add_argument('--weight_only', action='store_true', help='only load weight when resuming')
    parser.add_argument('--optimizer', default='adamw', help='Optimizer choice',choices=['adamw','adam','sgd'])
    parser.add_argument('--energy_save_mode', action='store_true', help='Only validate when needed')
    

    #save related
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='/mnt/team/t-yiqizhong/projects/ucf_results/', help='The path to the output log file')    
    parser.add_argument('--exp_name', default='', help='The name of the experiment')
    

    #loss related
    parser.add_argument('--loss_list', nargs="*", type=str, default=['recon'], help='loss type list')
    parser.add_argument('--eval_list', nargs="*", type=str, default=['psnr','ssim'], help='loss type list')
    parser.add_argument('--cos_restart', action='store_true', help='use cosine restart scheduler')
    parser.add_argument('--restart_ratio', default=0.5, type=float, help='lr drop ratio in cosine restart lr scheduler')
    parser.add_argument('--t_period', nargs="*", type=int,default=[], help='consine restart scheduler period')
    parser.add_argument('--only_val_last', action='store_true', help='Only evaluate last frame')

    
    args = parser.parse_args()
    print(args)
    torch.manual_seed(1024)
    cur_config = None
    if args.dataset.find('ucf')>-1:
        cur_config = configs.ucf_config
        from data.Dataloader import UCFPredDataset as PredDataset
    elif args.dataset.find('strpm')>-1:
        cur_config = configs.strpm_ucf_config
        from data.Dataloader import STRPM_UCFPredDataset as PredDataset
    elif args.dataset.find('kth') > -1:
        cur_config = configs.kth_config
        from data.Dataloader import KTHPredDataset as PredDataset
    elif args.dataset.find('mnist') > -1:
        cur_config = configs.mnist_config
        from data.Dataloader import MovingMNISTPredDataset as PredDataset
    
    cur_config['nepoch'] = args.nepoch
    cur_config['dataset'] = args.dataset
    cur_config['mode'] = args.mode
    cur_config['loss_list'] = args.loss_list
    cur_config['eval_list'] = args.eval_list
    cur_config['multistep'] = False
    cur_config['exp_name'] = args.exp_name
    cur_config['lr_step'] = args.step
    cur_config['decay_rate'] = args.decay
    cur_config['bilinear'] = args.bilinear
    cur_config['range'] = args.img_range
    cur_config['base_channel'] = args.base_channel
    cur_config['pred_base_channel'] = args.pred_base_channel
    cur_config['res_cat_img'] = args.res_cat_img
    cur_config['method'] = args.method
    cur_config['batch'] = args.batch
    cur_config['shuffle_setting'] = args.shuffle_setting
    cur_config['flip_aug'] = args.flip_aug
    cur_config['rot_aug'] = args.rot_aug
    cur_config['cos_restart'] = args.cos_restart
    cur_config['t_period'] = np.asarray(args.t_period)
    cur_config['add_feat_diff'] = args.add_feat_diff
    cur_config['filter_block'] = args.filter_block
    cur_config['res_img_scale'] = args.res_img_scale
    cur_config['val_subset'] = args.val_subset
    cur_config['pred_type'] = args.pred_type
    cur_config['pred_method'] = args.pred_method
    cur_config['use_bn'] = False
    cur_config['motion_use_bn'] = True
    cur_config['long_term'] = args.long_term
    cur_config['rrdb_encoder_num'] = args.rrdb_encoder_num
    cur_config['rrdb_decoder_num'] = args.rrdb_decoder_num
    cur_config['rrdb_enhance_num'] = args.rrdb_enhance_num
    cur_config['only_val_last'] = args.only_val_last
    cur_config['lr'] = args.lr
    cur_config['train_steps'] = 1 #placeholder
    cur_config['use_direct_predictor'] = args.use_direct_predictor
    cur_config['train_length'] = args.train_length
    cur_config['translate_factor'] = args.translate_factor
    cur_config['restart_ratio'] = args.restart_ratio
    cur_config['scale_in_use'] = args.scale_in_use
    if cur_config['base_channel'] // cur_config['translate_factor'] == 0 :
        print('translate_factor is too small! ')
        print('base_channel=',cur_cofig['base_channel'],' while translate_factor=',cur_config['translate_factor'])
        exit()
    if len(args.downsample_scale) > 0:
        cur_config['downsample_scale'] = args.downsample_scale

    if args.prev_len != -1:
        cur_config['prev_len'] = args.prev_len
    if args.fut_len != -1:
        cur_config['fut_len'] = args.fut_len
        cur_config['total_len'] = cur_config['fut_len'] + cur_config['prev_len']
    if args.long_term and args.fut_len == 1:
        print('Conflict! Long term mode while fut_len = 1')
        exit()

    # set mat size
    highres_scale = np.prod(cur_config['downsample_scale'][:-1])
    lowres_scale = np.prod(cur_config['downsample_scale'])
    if args.shuffle_setting:
        lowres_scale *= 2
        highres_scale *= 2
    
    
    cur_config['mat_size'] = [[cur_config['in_res'][0]//highres_scale,cur_config['in_res'][1]//highres_scale],[cur_config['in_res'][0]//lowres_scale,cur_config['in_res'][1]//lowres_scale]]
    print('Matrix size: ',cur_config['mat_size'])

    print('running config: ',cur_config)
    
    
    set_seed(1) # from SimVP
    
    main(cur_config,args)



