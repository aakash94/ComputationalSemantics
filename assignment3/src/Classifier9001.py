import torch
import torch.nn as nn
import torch.nn.functional as F

class TweetClassifier(nn.Module):
    def __init__(self, tweet_embedding_dime = 384, num_classes = 4):
        super(TweetClassifier, self).__init__()

        # 1 vector for reply and one for parent
        input_size = tweet_embedding_dime*2

        self.fc1  = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        # no activation used here because
        # https://stackoverflow.com/a/65193236

        return x