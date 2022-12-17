#Reference Page - https://github.com/endrol/DR_GCN/blob/9ad1929910ed30c3a623c25ba0da0198bd1655f5/dr_gcn/demo_dr_gcn.py
import argparse
from engine import *
from GCN import *
from dr import *
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings


parser = argparse.ArgumentParser(description='EyePacs Training')
parser.add_argument('--data', metavar='DIR',default='/home/sbx5057/Documents/COMP597/eyepacs_preprocess/eyepacs_preprocess',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate',default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_dr():
    warnings.filterwarnings('ignore')
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    #Reading the training labels
    csv_file = pd.read_csv('/home/sbx5057/Documents/COMP597/trainLabels.csv')

    #Taking 20% of the data for testing
    x,y = csv_file.iloc[:,0],csv_file.iloc[:,1]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


    #define datasets
    train_dataset = DRclassification(args.data,
                                     set='EyePacs Training Labels',train_data=x_train,label_data=y_train,
                                     inp_name='description/node.pkl',)
    val_dataset = DRclassification(args.data,
                                   set='EyePacs Testing Labels',train_data=x_test,label_data=y_test,
                                   inp_name='description/node.pkl')
    
                                   
    
    #GCN Model
    model = gcn_resnet101(num_classes=20, t=0.4, adj_file='description/edge.pkl')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size,'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':20}
    state['difficult_examples'] = True
   
    state['save_model_path'] = 'checkpoints/eyepacs/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

if __name__ == '__main__':
    main_dr()
  