import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out


class Generator448(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator448, self).__init__()
        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False) # out = (bn, 64, 224, 224)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)  # out = (bn, 128, 112, 112)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)  # out = (bn, 256, 56, 56)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)  # out = (bn, 512, 28, 28)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)  # out = (bn, 512, 14, 14)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8, padding=2)  # out = (bn, 512, 8, 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)  # out = (bn, 512, 4, 4)
        self.conv8 = ConvBlock(num_filter * 8, num_filter * 16)    # out = (bn, 1024, 2, 2)
        self.conv9 = ConvBlock(num_filter * 16, num_filter * 16, batch_norm=False)    # out = (bn, 1024, 1, 1)
        # Decoder, decode to 448x448
        self.deconv1 = DeconvBlock(num_filter * 16, num_filter * 16, dropout=True)    # out = (bn, 1024, 2, 2)
        self.deconv2 = DeconvBlock(num_filter * 16 * 2, num_filter * 8, dropout=True)   # out = (bn, 512, 4, 4)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)    # out = (bn, 512, 8, 8)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, padding=2)  # out = (bn, 512, 14, 14)
        self.deconv5 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)  # out = (bn, 512, 28, 28)
        self.deconv6 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)  # out = (bn, 256, 56, 56)
        self.deconv7 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)  # out = (bn, 128, 112, 112)
        self.deconv8 = DeconvBlock(num_filter * 2 * 2, num_filter)  # out = (bn, 64, 224, 224)
        self.deconv9 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)    # out = (bn, 3, 448, 448)
        
    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        enc8 = self.conv8(enc7)
        enc9 = self.conv9(enc8)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc9)
        dec1 = torch.cat([dec1, enc8], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc7], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc6], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc5], 1)
        dec5 = self.deconv5(dec4)
        dec5 = torch.cat([dec5, enc4], 1)
        dec6 = self.deconv6(dec5)
        dec6 = torch.cat([dec6, enc3], 1)
        dec7 = self.deconv7(dec6)
        dec7 = torch.cat([dec7, enc2], 1)
        dec8 = self.deconv8(dec7)
        dec8 = torch.cat([dec8, enc1], 1)
        dec9 = self.deconv9(dec8)
        out = torch.nn.Tanh()(dec9)
        
        return out
    
    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal_(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal_(m.deconv.weight, mean, std)
    

class Discriminator448(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator448, self).__init__()
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)   # (bn, 64, 224, 224)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)  # (bn, 128, 112, 112)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)  # (bn, 256, 56, 56)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=1)    # (bn, 512, 55, 55)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)  # (bn, 1, 54, 54)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal_(m.conv.weight, mean, std)