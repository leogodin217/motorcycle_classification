from torchvision import models
import torch.nn as nn
from torch import optim, cuda
import torch
from datetime import datetime
import sys


def get_model(num_classes, layers='34', generalization='batchnorm',
              dropout=0.4, classifier = nn.LogSoftmax(dim=1)):
    '''
    Creates a resnet-50 pretrained model and replaces the classifier with a new
    classifier
    num_classes: Number of classes in the data. Should be the same for train,
    valid and test
    generalization: Should we use batchnorm, dropout or none
    dropout: Proportion of features to drop out. Only used if generalization =
    'batchnorm'
    layers: 18, 34 or 50, 101 or 152 representing the resnet layers
    classifier: What classifier to classify the final output of the model
    '''
    if layers == '34':
        model = models.resnet34(pretrained=True)
    elif layers == '50':
        model = models.resnet50(pretrained=True)
    elif layers == '18':
        model = models.resnet18(pretrained=True)
    elif layers == '101':
        model = models.resnet101(pretrained=True)
    elif layers == '152':
        model = models.resnet152(pretrained=True)

    else:
        print('Invalid model')
        sys.exit(0)

    for param in model.parameters():
        param.requires_grad = False

    num_inputs = model.fc.in_features

    if generalization == 'batchnorm':
        print('Using batch normalization for generalization')
        model.fc = nn.Sequential(nn.Linear(num_inputs, 256),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(256),
                                 nn.Linear(256, num_classes),
                                 nn.BatchNorm1d(num_classes),
                                 classifier)
    elif generalization == 'dropout':
        print(f'Using dropout for generalization: dropout={dropout}')
        model.fc = nn.Sequential(nn.Linear(num_inputs, 256),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(256, num_classes),
                                 classifer)
    else:
        print(f'Using no generalization')
        model.fc = nn.Sequential(nn.Linear(num_inputs, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, num_classes),
                                 classifier)
    # Move to the GPU
    model = model.to('cuda')
    return model


def forward_pass(model, dataloader, criterion, num_images,
                 clear_cuda_cache=True):
    '''
    Performs a forward pass getting loss and accuracy, without modifying the
    model
    model: A pytorch NN
    data A pytorch Dataloader
    clear_cuda_cache: Do we want to clear cuda memory when possible?
    num_images: Int representing the number of images being processed.
                This is needed because a sampler might return fewer results
                than the total dataset.
    '''
    total_loss = 0

    with torch.no_grad():
        model.eval()

        for data, target in dataloader:
            data = data.to('cuda')
            target = target.to('cuda')
            result = model(data)
            loss = criterion(result, target)
            batch_loss = loss.item() * data.size(0)
            total_loss += batch_loss
            data = None
            target = None
            cuda.empty_cache()
    mean_loss = total_loss / num_images
    return({'mean_loss': mean_loss})


def get_accuracy(model, dataloader, num_images, topk=None):
    '''
    Performs a forward pass getting loss and accuracy, without modifying the
    model
    model: A pytorch NN
    data A pytorch Dataloader
    clear_cuda_cache: Do we want to clear cuda memory when possible?
    num_images: Int representing the number of images being processed. This is
                needed because a sampler might return fewer results than the
                total dataset.
    topk: Determines if we will use topk accuracy as well. If not, then no topk
          is used. If an int, k will be set to the int
    '''
    correct_predictions = 0
    correct_topk_predictions = 0
    with torch.no_grad():
        model.eval()
        for data, target in dataloader:
            data = data.to('cuda')
            target = target.to('cuda')
            result = model(data)
            _, predicted = torch.max(result.data, 1)
            # Get accurate images
            correct_predictions += (predicted == target).sum().item()
            # Get topk accuracy

            if topk is not None:
                top = result.data.topk(topk)[1]
                for index, top_k in enumerate(top):
                    if target[index] in top_k:
                        correct_topk_predictions += 1
            data = None
            target = None
            cuda.empty_cache()
    accuracy = {}
    accuracy['test_accuracy'] = correct_predictions / num_images
    if topk is not None:
        accuracy['test_accuracy_topk'] = correct_topk_predictions / num_images
    else:
        accuracy['test_accuracy_topk'] = 0

    return accuracy


def train_model(dataloaders, model, criterion, optimizer, epochs, topk=3,
                      clear_cuda_cache=True, name='basic model',
                      print_epoch=True):
    '''
    See https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    hyper parameters are set to the research recommendations
    Complete training function.
    dataloader: A Pytorch dataloader with train, validation and test datasets.
                All dataloaders should have the same number of classes
    clear_cuda_cache: Boolean telling us to clear the cuda cache when possible
    name: String with a name to give the model.
    topk: None or int specifying if the model should predict topk accuracy
    print_epoch: Bool specifying if themodel should print results each epoch
    '''
    all_start_time = datetime.now()
    results = []
    cuda_memory = []

    # The model changes with the number of classes, so we need to get that
    # number.
    # A dataset with a sampler, may not include all classes in the dataset, so
    # we need to iterate to find the distinct classes
    # We also need the number of images, with the same constraints as classes.
    # We will calculate it here for accuracy
    included_classes = []
    num_images = 0
    print('Gathering configurations')
    config_start_time = datetime.now()
    for item in dataloaders['train']:
        included_classes = included_classes + item[1].tolist()
        num_images += len(item[1])
    # Get the number of val images returned in the dataloader. It could be a
    # subset of the dataset if a sampler is used
    num_val_images = 0
    for item in dataloaders['val']:
        num_val_images += len(item[1])
    # Get the number of images returned in the dataloader. It could be a subset
    # of the dataset if a sampler s used
    num_test_images = 0
    for item in dataloaders['test']:
        num_test_images += len(item[1])
    config_end_time = datetime.now()
    print(f"Configuration: {config_end_time - config_start_time}")

    # Train the model
    model_start_time = datetime.now()
    for epoch in range(epochs):
        epoch_start = datetime.now()
        print(f'Epoch: {epoch + 1}')
        train_loss = 0.0
        for data, targets in dataloaders['train']:
            data = data.to('cuda')
            targets = targets.to('cuda')
            cuda_memory.append({
                'name': name,
                'timestamp': datetime.now(),
                'cuda_memory': cuda.memory_allocated()})
            # Clear the gradients
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, targets)
            loss.backward()
            # Get loss for the batch
            batch_loss = loss.item() * data.size(0)
            train_loss += batch_loss
            optimizer.step()
            # Clear the batch from cuda memory. It is no longer needed
            data = None
            targets = None
            cuda.empty_cache()
        mean_train_loss = train_loss / num_images

        # Get train accuracy to see if the model is learning at all
        # train_accuracy = get_accuracy(model, dataloaders['train'],
        # num_images)

        # Get validation loss
        validation_results = forward_pass(model, dataloaders['val'], criterion,
                                          num_images=num_val_images,
                                          clear_cuda_cache=clear_cuda_cache)
        # Get test accuracy
        test_accuracy = get_accuracy(model, dataloaders['test'],
                                     num_test_images, topk=topk)
        epoch_end = datetime.now()
        if print_epoch is True:
            print(f'Train_loss: {mean_train_loss}, Val loss: {validation_results["mean_loss"]}, Test Accuracy: {test_accuracy["test_accuracy"]}')
        results.append({
            'epoch': epoch + 1,
            'epoch_run_time': epoch_end - epoch_start,
            'train_loss': mean_train_loss,
            'val_loss': validation_results['mean_loss'],
            'test_accuracy': test_accuracy['test_accuracy'],
            'test_accuracy_topk': test_accuracy['test_accuracy_topk']})
    # Delete the model from cuda memory if needed
    if clear_cuda_cache is True:
        model = None
        cuda.empty_cache()
    model_end_time = datetime.now()

    all_end_time = datetime.now()
    return {
        'model': model,
        'name': name,
        'results': results,
        'run_time': all_end_time - all_start_time,
        'config_run_time': config_end_time - config_start_time,
        'model_run_time': model_end_time - model_start_time}


def get_adam_optimizer(model, lr=.001, betas=(0.9, 0.99), eps=10e-8,
                       weight_decay=0, amsgrad=False):
    '''
    Returns an ADAM optimizer with recommended defaults
    model: Torch NN model
    lr: (float, optional) – learning rate (default: 1e-3)
    betas: (Tuple[float, float], optional) – coefficients used for computing
           running averages of gradient and its square (default: (0.9, 0.999))
    eps: (float, optional) – term added to the denominator to improve numerical
         stability (default: 1e-8)
    weight_decay: (float, optional) – weight decay (L2 penalty) (default: 0)
    amsgrad: (boolean, optional) – whether to use the AMSGrad variant of this
             algorithm from the paper On the Convergence of Adam and Beyond
             (default: False)
    '''

    return(optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps,
                      weight_decay=weight_decay, amsgrad=amsgrad))


def get_sgd_optimizer(model, lr=0.001, momentum=0, weight_decay=0, dampening=0,
                      nesterov=False):
    '''
    Returns an SGD optimizer with recommended defaults
    model: A pytorch NN model
    lr: (float) – learning rate
    momentum: (float, optional) – momentum factor (default: 0)
    weight_decay: (float, optional) – weight decay (L2 penalty) (default: 0)
    dampening:(float, optional) – dampening for momentum (default: 0)
    nesterov: (bool, optional) – enables Nesterov momentum (default: False)
    '''
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                     weight_decay=weight_decay, dampening=dampening,
                     nesterov=nesterov)
