from __future__ import print_function, division
from flow_resnet import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.autograd import Variable
from makeDatasetFlow import *
import argparse
import sys

DEVICE = "cuda"

def main_run( trainDir, valDir, outDir, stackSize, trainBatchSize, valBatchSize, numEpochs, lr1,
             decay_factor, decay_step):


    ##if dataset == 'gtea61':
    num_classes = 61

    train_usr = ["S1", "S3", "S4"]
    val_usr = ["S2"]
    
    min_accuracy = 0

    model_folder = os.path.join('./', outDir, 'gtea61', 'flow')  # Dir for saving models and log files
    # Create the dir
    if os.path.exists(model_folder):
        print('Dir {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)

    # Log files
    writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')


    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])

    vid_seq_train = makeDataset(trainDir, train_usr, spatial_transform=spatial_transform, sequence=False,
                                stackSize=stackSize, fmt='.jpg')

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, sampler=None, num_workers=2, pin_memory=True)
    if valDir is not None:

        vid_seq_val = makeDataset(valDir, val_usr, spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                   sequence=False, stackSize=stackSize, fmt='.jpg', phase='Test')

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True)
        valInstances = vid_seq_val.__len__()


    trainInstances = vid_seq_train.__len__()
    print('Number of samples in the dataset: training = {} | validation = {}'.format(trainInstances, valInstances))

    model = flow_resnet34(True, channels=2*stackSize, num_classes=num_classes)
    model.train(True)
    train_params = list(model.parameters())

    #model.cuda()
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = torch.optim.SGD(train_params, lr=lr1, momentum=0.9, weight_decay=5e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step, gamma=decay_factor)

    train_iter = 0

    for epoch in range(numEpochs):
        optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.train(True)
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        for i, (inputs, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariable = Variable(inputs.to(DEVICE))
            labelVariable = Variable(targets.to(DEVICE))
            trainSamples += inputs.size(0)
            output_label, _ = model(inputVariable)
            loss = loss_fn(output_label, labelVariable)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.to(DEVICE)).sum()
            epoch_loss += loss.data[0]
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100
        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_loss, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch+1, avg_loss))
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch+1, trainAccuracy))
        if valDir is not None:
            if (epoch+1) % 1 == 0:
                model.train(False)
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                for j, (inputs, targets) in enumerate(val_loader):
                    val_iter += 1
                    val_samples += inputs.size(0)
                    inputVariable = Variable(inputs.to(DEVICE), volatile=True)
                    labelVariable = Variable(targets.to(DEVICE), volatile=True)
                    output_label, _ = model(inputVariable)
                    val_loss = loss_fn(output_label, labelVariable)
                    val_loss_epoch += val_loss.data[0]
                    _, predicted = torch.max(output_label.data, 1)
                    numCorr += (predicted == targets.to(DEVICE)).sum()
                val_accuracy = (numCorr / val_samples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Validation: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_flow_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_flow_state_dict_epoch' + str(epoch+1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)
    save_path_model = (model_folder + '/model_flow_state_dict_epoch' + str(epoch+1) + '.pth')              
    torch.save(model.state_dict(), save_path_model)
    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.close()


def __main__():
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    # parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
    #                     help='Train set directory')
    # parser.add_argument('--valDatasetDir', type=str, default=None,
    #                     help='Validation set directory')
    # parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    # parser.add_argument('--stackSize', type=int, default=5, help='Length of sequence')
    # parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    # parser.add_argument('--valBatchSize', type=int, default=32, help='Validation batch size')
    # parser.add_argument('--numEpochs', type=int, default=750, help='Number of epochs')
    # parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    # parser.add_argument('--stepSize', type=float, default=[150, 300, 500], nargs="+", help='Learning rate decay step')
    # parser.add_argument('--decayRate', type=float, default=0.5, help='Learning rate decay rate')

    # args = parser.parse_args()

    #dataset ='./GTEA61'
    trainDatasetDir = '/content/drive/MyDrive/ML_project/ego-rnn/content/GTEA61'
    valDatasetDir = '/content/drive/MyDrive/ML_project/ego-rnn/content/GTEA61'
    outDir ='results_flow'
    stackSize = 5
    trainBatchSize = 32
    valBatchSize = 32
    numEpochs = 750
    lr1 = 1e-2
    stepSize = [150, 300, 500]
    decayRate = 0.5

    main_run( trainDatasetDir, valDatasetDir, outDir, stackSize, trainBatchSize, valBatchSize, numEpochs, lr1,
             decayRate, stepSize)

__main__()
