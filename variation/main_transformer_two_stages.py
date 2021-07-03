from __future__ import print_function, division
from model_new_transformer import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from tensorboardX import SummaryWriter
from makeDatasetRGB import *
import argparse
import sys

DEVICE = "cuda"

def main_run( stage, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decay_factor, decay_step, memSize):
    
    train_usr = ["S1", "S3", "S4"]
    val_usr = ["S2"]

    num_classes = 61 # We are only using gtea61

    model_folder = os.path.join('./', out_dir, 'rgb', '-stage'+str(stage))  # Dir for saving models and log files
    # Create the dir
    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
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
    print(train_data_dir)

    vid_seq_train = makeDataset(train_data_dir,train_usr, spatial_transform, seqLen, True)

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=2, pin_memory=True) #ok

    #valuta
    if val_data_dir is not None:

        vid_seq_val = makeDataset(val_data_dir,val_usr, Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),seqLen,False)

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True)
        valInstances = vid_seq_val.__len__()

    train_params = []
    if stage == 1:

        model = selfAttentionModel(num_classes=num_classes, mem_size=memSize, num_frames=seqLen)
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
    #stage 2 : train anche per 
    else:
        model = selfAttentionModel(num_classes=num_classes, mem_size=memSize, num_frames=seqLen)
        model.load_state_dict(torch.load(stage1_dict), strict=True)
        model.train(False)
    
        for params in model.parameters():
            params.requires_grad = False
            #
        for params in model.resNet.layer4[0].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        
        for params in model.resNet.layer4[0].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[2].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
            
        for params in model.resNet.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
            
        for params in model.resNet.fc.parameters():
            params.requires_grad = True
            train_params += [params]
        
        model.resNet.layer4[0].conv1.train(True)
        model.resNet.layer4[0].conv2.train(True)
        model.resNet.layer4[1].conv1.train(True)
        model.resNet.layer4[1].conv2.train(True)
        model.resNet.layer4[2].conv1.train(True)
        model.resNet.layer4[2].conv2.train(True)
        model.resNet.fc.train(True)

      #Train the weights' matrices in the transofmer
    for params in model.transf.parameters():
        params.requires_grad = True
        train_params += [params]
      
      #Train the final classifier
    for params in model.fc.parameters():
        params.requires_grad = True
        train_params += [params]    

    model.transf.train(True)
    model.fc.train(True)
    
    model.cuda()



    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = torch.optim.SGD(train_params, lr=1e-3, weight_decay=4e-5, momentum=0.9)

    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fn, 10)

    train_iter = 0
    min_accuracy = 0

    for epoch in range(numEpochs):
        
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        if stage==2:
            model.resNet.layer4[0].conv1.train(True)
            model.resNet.layer4[0].conv2.train(True)
            model.resNet.layer4[1].conv1.train(True)
            model.resNet.layer4[1].conv2.train(True)
            model.resNet.layer4[2].conv1.train(True)
            model.resNet.layer4[2].conv2.train(True)
            model.resNet.fc.train(True)
        model.transf.train(True)
        model.fc.train(True)
        
        for i, (inputs, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            input_frames = Variable(inputs.to(DEVICE))
            ground_truth = Variable(targets.to(DEVICE))
            trainSamples += inputs.size(0)

            logits = model(input_frames)
            #print(logits.size())
            loss = loss_fn(logits, ground_truth)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(logits.data, 1)
            numCorrTrain += (predicted == targets.to(DEVICE)).sum()
            epoch_loss += loss.item()
          
        optim_scheduler.step()
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = (numCorrTrain.data.item() / trainSamples)

        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch+1, avg_loss, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1) # log del train
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        train_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, trainAccuracy))

        if val_data_dir is not None:
            if (epoch+1) % 1 == 0:
                model.train(False)
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                for j, (inputs, targets) in enumerate(val_loader):
                    val_iter += 1
                    val_samples += inputs.size(0)
                    inputVariable = Variable(inputs.to(DEVICE))
                    labelVariable = Variable(targets.to(DEVICE))
                    with torch.no_grad():
                      output_label = model(inputVariable)
                      val_loss = loss_fn(output_label, labelVariable)
                      val_loss_epoch += val_loss.item()
                    _, predicted = torch.max(output_label.data, 1)
                    numCorr += (predicted == targets.to(DEVICE)).sum()
                val_accuracy = (numCorr / val_samples)
                avg_val_loss = val_loss_epoch / val_iter
                print('Val: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_rgb_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_rgb_state_dict_epoch' + str(epoch+1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)
    
    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.close()


def __main__():
    stage = 1
    trainDatasetDir = './GTEA61'
    valDatasetDir = './GTEA61'
    stage1Dict = None
    outDir = 'results_stage1' # label for folder name
    seqLen = 7 # number of frames
    trainBatchSize = 32 # bnumber of training samples to work through before the model’s internal parameters are update
    valBatchSize = 32  # da valutare se 32 o 64
    numEpochs = 200 # 7 frame dovrebbe essere veloce
    lr1 = 1e-3 #defauld Learning rate
    decayRate = 0.1 #Learning rate decay rate
    stepSize = [25]
    memSize = 512 #ConvLSTM hidden state size


#Stage 1
    main_run(stage,
            trainDatasetDir,
            valDatasetDir,
            stage1Dict,
            outDir,
            seqLen, 
            trainBatchSize,
            valBatchSize,
            numEpochs, 
            lr1, 
            decayRate, 
            stepSize,
            memSize)    

    stage = 2
    trainDatasetDir = './GTEA61'
    valDatasetDir = './GTEA61'
    stage1Dict = './results_stage1/rgb/-stage1/model_rgb_state_dict.pth'
    outDir = 'results_stage2' # label for folder name
    seqLen = 7 # number of frames
    trainBatchSize = 32 # bnumber of training samples to work through before the model’s internal parameters are update
    valBatchSize = 32  # da valutare se 32 o 64
    numEpochs = 150 # 7 frame dovrebbe essere veloce
    lr1 = 1e-4 #defauld Learning rate
    decayRate = 0.1 #Learning rate decay rate
    stepSize = [25,75]
    memSize = 512 #ConvLSTM hidden state size


#Stage 2
    main_run(stage,
            trainDatasetDir,
            valDatasetDir,
            stage1Dict,
            outDir,
            seqLen, 
            trainBatchSize,
            valBatchSize,
            numEpochs, 
            lr1, 
            decayRate, 
            stepSize,
            memSize)

__main__()
