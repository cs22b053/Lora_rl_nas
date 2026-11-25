# eval_best.py  — run from /scratch/ajin/temp/NAO_V2_1/
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import utils, logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S %p')

CKPT_BEST = 'exp_v2/NAONet_V2_test_v0/checkpoint_best.pt'
CKPT_SWA  = 'exp_v2/NAONet_V2_test_v0/checkpoint_swa.pt'
DATA   = 'data'

from model import NASNetworkCIFAR

best_state = torch.load(CKPT_BEST, map_location='cpu')
swa_state = torch.load(CKPT_SWA, map_location='cpu')

args  = best_state['args']
logging.info('SWA checkpoint: epoch=%d  swa_acc_recorded=%.4f%%',
             swa_state['epoch'], swa_state['swa_acc'])

cudnn.benchmark = False
cudnn.deterministic = True

_, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
test_data = dset.CIFAR10(root=DATA, train=False, download=True, transform=valid_transform)
test_queue = torch.utils.data.DataLoader(
    test_data, batch_size=500, shuffle=False, pin_memory=True, num_workers=8)

model = NASNetworkCIFAR(args, 10, args.layers, args.nodes, args.channels,
                        args.keep_prob, args.drop_path_keep_prob,
                        args.use_aux_head, args.steps, args.arch)
model.load_state_dict(swa_state['model'])
model = model.cuda()
model.eval()

criterion = nn.CrossEntropyLoss().cuda()
top1 = utils.AvgrageMeter()
top5 = utils.AvgrageMeter()

with torch.no_grad():
    for step, (x, y) in enumerate(test_queue):
        x, y = x.cuda(), y.cuda()
        logits, _ = model(x)
        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

logging.info('==> Test top-1: %.4f%%   top-5: %.4f%%', top1.avg, top5.avg)