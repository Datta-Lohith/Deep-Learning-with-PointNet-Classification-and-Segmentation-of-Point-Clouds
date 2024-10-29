import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, global_feat=False):
        super(PointNet, self).__init__()
        self.Conv_1 = torch.nn.Conv1d(3, 64, 1)
        self.Conv_2 = torch.nn.Conv1d(64, 128, 1)
        self.Conv_3 = torch.nn.Conv1d(128, 1024, 1)
        self.BNorm_1 = nn.BatchNorm1d(64)
        self.BNorm_2 = nn.BatchNorm1d(128)
        self.BNorm_3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, out):
        out = out.transpose(2, 1)
        out = F.relu(self.BNorm_1(self.Conv_1(out)))
        local_feat = out
        out = F.relu(self.BNorm_2(self.Conv_2(out)))
        out = self.BNorm_3(self.Conv_3(out))
        out = torch.max(out, 2, keepdim=True)[0]
        out= out.view(-1, 1024)
        if self.global_feat:
            return out
        else:
            out = out.unsqueeze(2).repeat(1, 1, local_feat.shape[2])
            out = torch.cat([out, local_feat], 1)
            return out
        

class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.PNet = PointNet(global_feat=True)
        self.fConv_1 = nn.Linear(1024, 512)
        self.fConv_2 = nn.Linear(512, 256)
        self.fConv_3 = nn.Linear(256, num_classes)
        self.BNorm_1 = nn.BatchNorm1d(512)
        self.BNorm_2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        out = self.PNet(points)
        out = F.relu(self.BNorm_1(self.fConv_1(out)))
        out = F.relu(self.BNorm_2(self.drop(self.fConv_2(out))))
        out = self.fConv_3(out)
        out = F.log_softmax(out, dim=1)
        return out



class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.num_seg_classes = num_seg_classes
        self.PNet = PointNet()
        self.Conv_1 = torch.nn.Conv1d(1088, 512, 1)
        self.Conv_2 = torch.nn.Conv1d(512, 256, 1)
        self.Conv_3 = torch.nn.Conv1d(256, 128, 1)
        self.Conv_4 = torch.nn.Conv1d(128, self.num_seg_classes, 1)
        self.BNorm_1 = nn.BatchNorm1d(512)
        self.BNorm_2 = nn.BatchNorm1d(256)  
        self.BNorm_3 = nn.BatchNorm1d(128)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        B, N, _= points.size()
        out = self.PNet(points)
        out = F.relu(self.BNorm_1(self.Conv_1(out)))
        out = F.relu(self.BNorm_2(self.Conv_2(out)))
        out = F.relu(self.BNorm_3(self.Conv_3(out)))
        out = self.Conv_4(out)
        out = out.transpose(2, 1).contiguous()
        out = out.view(-1, self.num_seg_classes)
        out = F.log_softmax(out, dim=-1)
        out = out.view(B, N, self.num_seg_classes)
        return out



