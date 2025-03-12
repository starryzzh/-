from recbole.model.context_aware_recommender import BPR

class RecBoleMFModel(BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # 配置矩阵分解参数
        self.embedding_size = config['embedding_size']
        self.reg_weights = config['reg_weights']
        
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        
        # 计算BPR损失
        user_e = self.user_embedding(user)
        pos_e = self.item_embedding(pos_item)
        neg_e = self.item_embedding(neg_item)
        
        pos_score = (user_e * pos_e).sum(dim=1)
        neg_score = (user_e * neg_e).sum(dim=1)
        
        loss = -((pos_score - neg_score).sigmoid().log().mean())
        loss += self.get_regularization_loss()
        
        return loss