import torch
import torch.nn as nn

import os



class Conv_decoder(nn.Module):
    def __init__(self):
        super(Conv_decoder, self).__init__()

        # Two fully connected layers of neurons (feedforward architecture)
        self.ff_layers=nn.Sequential( 
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 7 * 10 * 16),  # 1120 neurons
            nn.ReLU(),
        )

        # Sequential upsampling using the deconvolutional layers & smoothing out checkerboard artifacts with conv layers
        self.conv_layers=nn.Sequential(
            nn.ConvTranspose2d(16, 64, 4, stride=2, padding=1), #deconv1
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), #conv1
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1), #deconv2
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1), #conv2
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), #deconv3
            nn.Sigmoid() #Squeezing the output to 0-1 range
        )


    def forward(self, x):
        x = self.ff_layers(x)        
        x = x.view(-1, 16, 7, 10) # Reshaping the output of the fully connected layers so that it is compatible with the conv layers
        x = self.conv_layers(x)
        return x

if __name__ == '__main__':
    decoder_net = Conv_decoder()
    decoder_net.load_state_dict(torch.load(os.path.join('Trained_Model','trained_decoder_net')))
    decoder_net.eval()
    print("successfully loaded model!")