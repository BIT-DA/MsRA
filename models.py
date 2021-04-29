from utils import *
from Multi_spectral_resnet import *


class MsRa_model(nn.Module):
    def __init__(self):
        super(MsRa_model, self).__init__()
        self.feature = ms_resnet50(pretrained=True)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(256, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 1))
        self.class_classifier.add_module('c_sigmoid',nn.Sigmoid())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        x = self.feature(input_data)
        x = x.view(-1, 256)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        class_output = self.class_classifier(x)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output,x


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = MsRa_model().to(device)
    input_s = torch.randn(2, 3, 224, 224).to(device)
    class_output, domain_output,feature = m(input_s,0.1)
    print(list(feature.size()))
