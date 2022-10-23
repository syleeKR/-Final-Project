import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class BetterNet(nn.Module):
    def __init__(self):
        super(BetterNet, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=1,
                               groups=1, bias=True, padding_mode='zeros')
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=1,
                                groups=1, bias=True, padding_mode='zeros')

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception1 = Inception(32, 16, 24, 32, 4, 8, 8)
        self.inception2 = Inception(64, 32, 32, 48, 8, 24, 16)
        self.inception3 = Inception(120, 48, 24, 52, 4, 12, 16)
        self.inception4 = Inception(128, 40, 28, 56, 6, 16, 16)
        self.inception5 = Inception(128, 32, 32, 64, 6, 16, 16)

        self.fc3 = nn.Linear(2048, 50)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, x):
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        x= torch.flatten(x, start_dim=0, end_dim=1)


        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv11(x)))
        x = self.pool2(x)

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool2(x)

        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.fc3(x)

        out = x
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return out
'''
class CustomCNN(nn.Module):
    def __init__(self):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-1: define cnn model
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                               bias=True, padding_mode='zeros')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2 = nn.Conv2d(8, 16, 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 40)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        For reference (shape example)
        inputs: Batch size X (Sequence_length, Channel=1, Height, Width)
        outputs: (Sequence_length X Batch_size, Hidden_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem1-2: code CNN forward path

        inputs = torch.flatten(inputs, start_dim=0, end_dim=1)
        x = self.pool1(F.relu(self.conv1(inputs)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return x
'''
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, vocab_size, num_layers, dropout):
        super(LSTM, self).__init__()

        # define the properties
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2-1: Define lstm and input, output projection layer to fit dimension
        # output fully connected layer to project to the size of the class
        
        # you can either use torch LSTM or manually define it
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_in = nn.Linear(input_dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, feature, h, c):
        """
        For reference (shape example)
        feature: (Sequence_length, Batch_size, Input_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem2-2: Design LSTM model for letter sorting
        # NOTE: sequence length of feature can be various
        feature = self.fc_in(feature)
        output, hidden = self.lstm(feature, (h, c))
        h_next, c_next = hidden
        output = self.fc_out(output)
        output = self.softmax(output)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        
        # (sequence_lenth, batch, num_classes), (num_rnn_layers, batch, hidden_dim), (num_rnn_layers, batch, hidden_dim)
        return output, h_next, c_next  


class ConvLSTM(nn.Module):
    def __init__(self, sequence_length=None, num_classes=26, cnn_layers=None,
                 cnn_input_dim=1, rnn_input_dim=50,
                 cnn_hidden_size=256, rnn_hidden_size=100, rnn_num_layers=1, rnn_dropout=0.2):
        # NOTE: you can freely add hyperparameters argument
        super(ConvLSTM, self).__init__()

        # define the properties, you can freely modify or add hyperparameters
        #self.cnn_hidden_size = cnn_hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        #self.cnn_input_dim = cnn_input_dim
        self.rnn_input_dim = rnn_input_dim
        self.rnn_num_layers = rnn_num_layers
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        #self.conv = CustomCNN()
        self.lstm = LSTM(input_dim=rnn_input_dim, hidden_size=rnn_hidden_size, vocab_size=num_classes, num_layers=rnn_num_layers,dropout=rnn_dropout)
        self.betterconv = BetterNet()
        # NOTE: you can define additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs):
        """
        input is (imgaes, labels) (training phase) or images (test phase)
        images: sequential features of Batch size X (Sequence_length, Channel=1, Height, Width)
        labels: Batch size X (Sequence_length)
        outputs should be a size of Batch size X (1, Num_classes) or Batch size X (Sequence_length, Num_classes)
        """

        # for teacher-forcing
        have_labels = False
        if len(inputs) == 2:
            have_labels = True
            images, labels = inputs
        else:
            images = inputs

        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem3: input image into CNN and RNN sequentially.
        # NOTE: you can use teacher-forcing using labels or not
        # NOTE: you can modify below hint code

        tensorlist =[]
        
        maxlen =2
        batchsize =0
        for data in images:
            batchsize+=1
            maxlen = max(maxlen , int(data.size()[0]) )
        a = [[[[0 for col in range(28)] for row in range(28)] for depth in range(1)] for k in range(1)]
        a = torch.tensor(a)



        for data in images:
            deficit = maxlen - int(data.size()[0])
            for i in range(deficit):
                data = torch.cat([a, data], 0)
            tensorlist.append(data)

        images = torch.stack(tensorlist, 0).cuda()
        self.sequence_length = maxlen


        '''
        if have_labels:
            for i  in range(len(labels)):
                deficit = self.sequence_length - (len(labels[i])-1)
                a = [26]*deficit
                a = torch.tensor(a)

                labels[i] = torch.cat([a,labels[i]],0)
        '''


        hidden_state = torch.zeros(1, batchsize, self.rnn_hidden_size, device="cuda")
        cell_state = torch.zeros(1, batchsize, self.rnn_hidden_size, device="cuda")


        output_of_conv = self.betterconv(images)
        output_of_conv = torch.reshape(output_of_conv, (batchsize, self.sequence_length, self.rnn_input_dim))
        outputs, _, _ = self.lstm(output_of_conv, hidden_state, cell_state)



        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs


