import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import inspect
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

def show_models():
    """ prints the name of the class of all the available autoencoder models """
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    models = [m[0] for m in clsmembers if m[0] != 'Autoencoder' and m[0] != 'Variable']
    print(models)

def get_model(name):
    """ prints the class, the size of the trajectories and the size of the latent space of the model saved in 'name'
    :return: list of int, [length trajectories, latent space, average pooling size]
    """
    model_dict = torch.load(name, map_location='cpu')
    print('autoencoder model: ' + model_dict['autoencoder class'])
    print('length of the trajectories: ' + str(model_dict['length trajectories']))
    print('size of the latent space: ' + str(model_dict['latent space']))
    print('size average pooling: ' + str(model_dict['average pooling size']))
    return([model_dict['length trajectories'], model_dict['latent space'], model_dict['average pooling size']])

class Autoencoder(nn.Module):
    def __init__(self, size_trajectory, latent_space, avg_pooling, decode = True):
        super(Autoencoder, self).__init__()
        """ an autoencoder object is created with atributs for the size of the trajectories that will be used to train it,
            the size of the latent space, and a parameter called decode which is automatically set to True. If decode=False,
            the object will return the output of the encoder when a trajectory is inputted
        """
        self.size_trajectory = size_trajectory
        self.latent_space = latent_space
        self.avg_pooling = avg_pooling
        self.decode = decode
        self.training_loss_per_epoch = np.array([])
        self.validation_loss_per_epoch = np.array([])
        self.epochs = 0
    
    def train_model(self, epochs, dataloader, distance, optimizer, device, testloader = None):
        """ trains the model, updates the weights and biases automatically. At each epoch the loss is added
        into the train loss array
        """
        total_epochs = self.epochs + epochs
        
        for epoch in range(epochs):
            # Train data
            loss_epoch, dataset_size = (0,0)
            for data in dataloader:
                img = Variable(data).to(device)
                # forward
                output = self(img)
                loss = distance(output, img)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item() * len(data)
                dataset_size += len(data)
            
            self.training_loss_per_epoch = np.append(self.training_loss_per_epoch, loss_epoch/dataset_size)
            # the size of the data set is not always a multiple of the batch size, therefore we need to take
            # into account the size of the batch that we load each time since the last batch may be smaller.
            # That's why we multiply each time by len(data) and we divide by the size of the data set in the end
            
            if testloader:
                val_loss_epoch, valset_size = (0,0)
                for data in testloader:
                    img = Variable(data).to(device)
                    loss = distance(self(img), img)
                    val_loss_epoch += loss.item() * len(data)
                    valset_size += len(data)
                
                self.validation_loss_per_epoch = np.append(self.validation_loss_per_epoch, val_loss_epoch/valset_size)
                    
            self.epochs += 1
            
            if self.epochs % 5 == 0:
                print('\nepoch [{}/{}], training loss:{:.4f}'.format(self.epochs, total_epochs, loss_epoch/dataset_size), end='')
                if self.validation_loss_per_epoch.any():
                    print(', validation loss:{:.4f}'.format(val_loss_epoch/valset_size), end='')
 
        print('\nmodel training completed')

    def test_model(self, dataloader, distance, device):
        #test_loss, iterations = (0,0)
        test_loss = 0
        for i, data in enumerate(dataloader):
            img = data.to(device)
            output = self(img)
            test_loss += distance(output, img)
            #iterations += 1
        test_loss /= (i+1)
        return test_loss

    @staticmethod
    def get_output_size(input_size, kernel_size, stride, padding):
        """
        :return: int, size of the output tensor of a convolutional or max pooling layer given the input size
        """
        output_size = int((input_size - kernel_size + 2*padding)/stride + 1)
        return output_size
    
    def calculate_layers_output_size(self, iterations, size, kernel_size, stride, padding):
        """ generator to calculate the output of multiple layers with the same kernel size, stride and padding
        :yield: int, size of the output tensor of a convolutional or max pooling layer given the input size
        """
        for i in range (iterations):
            size = self.get_output_size(size, kernel_size, stride, padding)
            yield size
    
    def get_layers_output_size(self, n_layers, input_size, kernel_size, stride, padding=0):
        """
        :return: list of int, size of the output tensors of multiple convolutional or max pooling layers with the same
                 kernel size, stride and padding given the size of the input tensor in the first layer
        """
        output_generator = self.calculate_layers_output_size(n_layers, input_size, kernel_size, stride, padding)
        output = [i for i in output_generator]
        return output
        
    def set_ConvTranspose1d(self, input_size, output_size, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        :return: nn.ConvTranspose1d with adjusted parameters according to the target output size given the input size
        """
        kernel_size, padding = self.adjust_transpose_layer(input_size, output_size, kernel_size, stride, padding)
        return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def set_MaxUnpool1d(self, input_size, output_size, kernel_size, stride, padding=0):
        """
        :return: nn.MaxUnpool1d with adjusted parameters according to the target output size given the input size
        """
        kernel_size, padding = self.adjust_transpose_layer(input_size, output_size, kernel_size, stride, padding)
        return nn.MaxUnpool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    
    @staticmethod
    def adjust_transpose_layer(input_size, output_size, kernel_size, stride, padding):
        """ method for adjusting the kernel size and padding of a convolutional transpose or maxunpooling layer
            according to the target output size given the input size
            :return: int, int, kernel size, padding
        """
        output = (input_size-1)*stride-2*padding+kernel_size
        difference = output - output_size
        if difference < 0:
            kernel_size -= difference
            difference = 0
        elif difference%2 != 0:
            kernel_size -= 1
            difference -= 1
        padding += int(difference/2)
        return kernel_size, padding
    
    def double_conv(in_channels, out_channels):
        """
        :return: double 3x3 convolution layer with stride and padding of 1
        """
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def get_number_of_parameters(self):
        """
        :return: int, number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters())
    
    def save_model(self, parameters, name):
        """ saves a dictionary in disk with the model state, its atributes and any training specifications defined in parameters
        """
        save_dict = {'model': self.state_dict(), 
                     'autoencoder class': self.__class__.__name__,
                     'length trajectories': self.size_trajectory,
                     'latent space': self.latent_space,
                     'average pooling size': self.avg_pooling,
                     'epochs': self.epochs}
        save_dict.update(parameters)
        save_dict['training loss'] = self.training_loss_per_epoch
        save_dict['validation loss'] = self.validation_loss_per_epoch
        torch.save(save_dict, name)
        print('model saved')
    
    def load_model(self, name, cuda=False, evaluate=False):
        """ loads a trained model and a dictionary with all the saved training parameters
        """
        device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        self.model_dict = torch.load(name, map_location = device)
        self.load_state_dict(self.model_dict['model'])
        self.training_loss_per_epoch = self.model_dict['training loss']
        self.validation_loss_per_epoch = self.model_dict['validation loss']
        self.epochs = self.model_dict['epochs']
        if evaluate:
            self.eval()
        return('model loaded')
    
    def plot_loss(self, figsize=(8,6.5), logscale=True, split='train'):
        """ plots the loss-per-epoch of a trained model
        """
        if not self.training_loss_per_epoch.any():
            print('The model is not trained, there is no training loss to plot')
        else:
            plt.figure(figsize=figsize)
            if split == 'train':
                loss = self.training_loss_per_epoch
                title = 'Training loss'
            elif split == 'validation':
                loss = self.validation_loss_per_epoch
                title = 'Validation loss'
            else:
                print('choose split="train" or split="validation"')
            if logscale:
                plt.title(title + ' (log scale)', fontsize=28, y=1.02)
                plt.loglog(loss, c='r')
            else:
                plt.title(title + ' (normal scale)', fontsize=28, y=1.02)
                plt.plot(loss, c='r')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('epoch', fontsize=20)
            plt.ylabel('loss', fontsize=20)
    
    def show_model_properties(self):
        """ shows the training hyperparameters and any other stored model specifications
        """
        if self.training_loss_per_epoch.any():
            excludes = ['model', 'training loss', 'validation loss', 'optimizer state dict']
            keys = set(self.model_dict.keys())
            print('\033[1m'+'Training hyperparameters & diffusion models specifications:'+'\033[0m')
            for key in keys.difference(excludes):
                print(f'{key}: ' + str(self.model_dict[key]))
            print('minimum training loss: ' + str(np.min(self.training_loss_per_epoch)))
            if self.validation_loss_per_epoch.any():
                print('minumum validation loss: ' + str(np.min(self.validation_loss_per_epoch)))
                print('best epoch (validation set): ' + str(np.argmin(self.validation_loss_per_epoch) + 1))
        else: print ('The model is not trained, there are no hyperparameters to show')

    
class Final_Skip_Connections(Autoencoder):
    def __init__(self, size_trajectory, latent_space=20, avg_pooling=4, encoder_channels_out=64):
        super(Final_Skip_Connections, self).__init__(size_trajectory = size_trajectory, latent_space = latent_space, avg_pooling = avg_pooling)
        
        self.encoder_channels_out = encoder_channels_out
        
        self.conv0 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv6 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(64, encoder_channels_out, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        self.globalAvg = nn.AdaptiveAvgPool1d(avg_pooling)
        self.fc1 = nn.Linear(encoder_channels_out*avg_pooling, 96)
        self.fc2 = nn.Linear(96, latent_space)
        self.fc3 = nn.Linear(latent_space, 96)
        self.fc4 = nn.Linear(96, encoder_channels_out*int(size_trajectory/8))
        self.convtranspose0 = nn.ConvTranspose1d(64, 64, kernel_size=3, padding=1)
        self.convtranspose1 = nn.ConvTranspose1d(64, 64, kernel_size=3, padding=1)
        self.convtranspose2 = nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2)
        self.convtranspose3 = nn.ConvTranspose1d(32, 32, kernel_size=3, padding=1)
        self.convtranspose4 = nn.ConvTranspose1d(32, 32, kernel_size=3, padding=1)
        self.convtranspose5 = nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2)
        self.convtranspose6 = nn.ConvTranspose1d(16, 16, kernel_size=3, padding=1)
        self.convtranspose7 = nn.ConvTranspose1d(16, 16, kernel_size=3, padding=1)
        self.convtranspose8 = nn.ConvTranspose1d(16, 1, kernel_size=5, padding=2)
        self.maxunpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
        
    def forward(self,x):
        x = torch.relu(self.conv0(x))
        identity = x
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x) + identity)
        x = self.bn1(x)
        x, idx1 = self.pool(x)
        x = torch.relu(self.conv3(x))
        identity = x
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x) + identity)
        x = self.bn2(x)
        x, idx2 = self.pool(x)
        x = torch.relu(self.conv6(x))
        identity = x
        x = torch.relu(self.conv7(x))
        x = torch.relu(self.conv8(x) + identity)
        x = self.bn3(x)
        x, idx3 = self.pool(x)
        x = self.globalAvg(x)
        x = x.view(-1, 1, self.encoder_channels_out*self.avg_pooling)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        if self.decode:
            x = self.fc3(x)
            x = self.fc4(x)
            x = x.view(-1, self.encoder_channels_out, int(self.size_trajectory/8))
            x = self.bn3(x)
            x = self.maxunpool(x, idx3)
            x = self.convtranspose0(x)
            x = self.convtranspose1(x)
            x = self.convtranspose2(x)
            x = self.bn2(x)
            x = self.maxunpool(x, idx2)
            x = self.convtranspose3(x)
            x = self.convtranspose4(x)
            x = self.convtranspose5(x)
            x = self.bn1(x)
            x = self.maxunpool(x, idx1)
            x = self.convtranspose6(x)
            x = self.convtranspose7(x)
            x = self.convtranspose8(x)
            
        return (x)
    
    
class Small(Autoencoder):
    def __init__(self, size_trajectory, latent_space=4, avg_pooling=1, encoder_channels_out=64):
        super(Small, self).__init__(size_trajectory = size_trajectory, latent_space = latent_space, avg_pooling = avg_pooling)
        
        self.encoder_channels_out = encoder_channels_out
            
        #self.dropout = nn.Dropout()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv6  = nn.Conv1d(64, encoder_channels_out, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        self.global_pool = nn.AdaptiveAvgPool1d(self.avg_pooling)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(avg_pooling*encoder_channels_out, 32)
        self.fc2 = nn.Linear(32, latent_space)
        self.fc3 = nn.Linear(latent_space, 32)
        self.fc4 = nn.Linear(32, encoder_channels_out*int(size_trajectory/4))
        self.deconv1 = nn.ConvTranspose1d(encoder_channels_out, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose1d(32, 32, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2)
        self.deconv5 = nn.ConvTranspose1d(16, 16, kernel_size=3, padding=1)
        self.deconv6 = nn.ConvTranspose1d(16, 1, kernel_size=5, padding=2)
        self.maxunpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
            
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x, idx1 = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x, idx2 = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = self.global_pool(x)
        x = x.view(-1, 1, self.encoder_channels_out*self.avg_pooling)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.decode:
            x = self.fc3(x)
            x = self.fc4(x)
            x = x.view(-1, self.encoder_channels_out, int(self.size_trajectory/4))
            x = self.bn3(x)
            x = self.deconv1(x)
            x = self.deconv2(x)
            x = self.maxunpool(x, idx2)
            x = self.bn2(x)
            x = self.deconv3(x)
            x = self.deconv4(x)
            x = self.maxunpool(x, idx1)
            x = self.bn1(x)
            x = self.deconv5(x)
            x = self.deconv6(x)
            
        return x