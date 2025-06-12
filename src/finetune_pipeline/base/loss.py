import torch


class TripletMarginLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, query_emb, pos_emb, neg_emb):
        # L2 거리 계산
        pos_dist = torch.norm(query_emb - pos_emb, p=2, dim=1)
        neg_dist = torch.norm(query_emb - neg_emb, p=2, dim=1)

        # max(0, pos_dist - neg_dist + margin) 형태의 손실
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()