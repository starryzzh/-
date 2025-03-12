try:
    from recbole.model.context_aware_recommender import ItemKNN
except ImportError:
    import subprocess
    try:
        print("尝试安装RecBole库...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "recbole"])
        from recbole.model.context_aware_recommender import ItemKNN
        print("RecBole库安装成功，已成功导入ItemKNN。")
    except subprocess.CalledProcessError:
        print("无法解析导入'recbole.model.context_aware_recommender'。自动安装RecBole库失败，请手动检查并安装。")
        import sys
        sys.exit(1)

class RecBoleCFModel(ItemKNN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # 设置协同过滤超参数
        self.k = config['k_neighbors']
        self.shrink = config['shrink_factor']
        
    def calculate_loss(self, interaction):
        # 实现基于物品的协同过滤
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        # 计算相似度矩阵
        sim_matrix = self.similarity_matrix
        
        # 获取用户历史交互
        user_inter = self.user_interaction_matrix[user]
        
        # 生成预测评分
        pred = (user_inter @ sim_matrix) / (self.shrink + user_inter.sum(axis=1))
        
        return self.loss_fn(pred, interaction[self.RATING])