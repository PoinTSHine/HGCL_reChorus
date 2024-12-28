### HGCL
# -*- coding: UTF-8 -*
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from models.BaseModel import SequentialModel

### HGCL
class MODEL(nn.Module):
    def __init__(self,args, userNum, itemNum, userMat, itemMat, uiMat, hide_dim, Layers):
        super(MODEL, self).__init__()
        self.args = args
        self.userNum = userNum
        self.itemNum = itemNum
        self.uuMat = userMat
        self.iiMat = itemMat
        self.uiMat = uiMat
        self.hide_dim = hide_dim
        self.LayerNums = Layers
        
        uimat   = self.uiMat[: self.userNum,  self.userNum:]
        values  = torch.FloatTensor(uimat.tocoo().data)
        indices = np.vstack(( uimat.tocoo().row,  uimat.tocoo().col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape =  uimat.tocoo().shape
        uimat1=torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.uiadj = uimat1
        self.iuadj = uimat1.transpose(0,1)
        
        self.gating_weightub=nn.Parameter(
            torch.FloatTensor(1,hide_dim))
        nn.init.xavier_normal_(self.gating_weightub.data)
        self.gating_weightu=nn.Parameter( 
            torch.FloatTensor(hide_dim,hide_dim))
        nn.init.xavier_normal_(self.gating_weightu.data)
        self.gating_weightib=nn.Parameter( 
            torch.FloatTensor(1,hide_dim))
        nn.init.xavier_normal_(self.gating_weightib.data)
        self.gating_weighti=nn.Parameter(
            torch.FloatTensor(hide_dim,hide_dim))
        nn.init.xavier_normal_(self.gating_weighti.data)

        self.encoder = nn.ModuleList()
        for i in range(0, self.LayerNums):
            self.encoder.append(GCN_layer())
        self.k = args.rank 
        k = self.k
        self.mlp  = MLP(hide_dim,hide_dim*k,hide_dim//2,hide_dim*k)
        self.mlp1 = MLP(hide_dim,hide_dim*k,hide_dim//2,hide_dim*k)
        self.mlp2 = MLP(hide_dim,hide_dim*k,hide_dim//2,hide_dim*k)
        self.mlp3 = MLP(hide_dim,hide_dim*k,hide_dim//2,hide_dim*k)
        self.meta_netu = nn.Linear(hide_dim*3, hide_dim, bias=True)
        self.meta_neti = nn.Linear(hide_dim*3, hide_dim, bias=True)

        self.embedding_dict = nn.ModuleDict({
        'uu_emb': torch.nn.Embedding(userNum, hide_dim).cuda(),
        'ii_emb': torch.nn.Embedding(itemNum, hide_dim).cuda(),
        'user_emb': torch.nn.Embedding(userNum , hide_dim).cuda(),
        'item_emb': torch.nn.Embedding(itemNum , hide_dim).cuda(),
        })

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
        })
        return embedding_dict
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    def metaregular(self,em0,em,adj):
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[:,torch.randperm(embedding.shape[1])]
            corrupted_embedding = corrupted_embedding[torch.randperm(embedding.shape[0])]
            return corrupted_embedding
        def score(x1,x2):
            x1=F.normalize(x1,p=2,dim=-1)
            x2=F.normalize(x2,p=2,dim=-1)
            return torch.sum(torch.multiply(x1,x2),1)
        user_embeddings = em
        Adj_Norm =t.from_numpy(np.sum(adj,axis=1)).float().cuda()
        adj=self.sparse_mx_to_torch_sparse_tensor(adj)
        edge_embeddings = torch.spmm(adj.cuda(),user_embeddings)/Adj_Norm
        user_embeddings=em0
        graph = torch.mean(edge_embeddings,0)
        pos   = score(user_embeddings,graph)
        neg1  = score(row_column_shuffle(user_embeddings),graph)
        global_loss = torch.mean(-torch.log(torch.sigmoid(pos-neg1)))
        return global_loss 

    def self_gatingu(self,em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em,self.gating_weightu) + self.gating_weightub))
    def self_gatingi(self,em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em,self.gating_weighti) + self.gating_weightib))

    def metafortansform(self, auxiembedu,targetembedu,auxiembedi,targetembedi):
       
        # Neighbor information of the target node
        uneighbor=t.matmul( self.uiadj.cuda(),self.ui_itemEmbedding)
        ineighbor=t.matmul( self.iuadj.cuda(),self.ui_userEmbedding)

        # Meta-knowlege extraction
        tembedu=(self.meta_netu(t.cat((auxiembedu,targetembedu,uneighbor),dim=1).detach()))
        tembedi=(self.meta_neti(t.cat((auxiembedi,targetembedi,ineighbor),dim=1).detach()))
        
        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        metau1=self.mlp( tembedu). reshape(-1,self.hide_dim,self.k)# d*k
        metau2=self.mlp1(tembedu). reshape(-1,self.k,self.hide_dim)# k*d
        metai1=self.mlp2(tembedi). reshape(-1,self.hide_dim,self.k)# d*k
        metai2=self.mlp3(tembedi). reshape(-1,self.k,self.hide_dim)# k*d
        meta_biasu =(torch.mean( metau1,dim=0))
        meta_biasu1=(torch.mean( metau2,dim=0))
        meta_biasi =(torch.mean( metai1,dim=0))
        meta_biasi1=(torch.mean( metai2,dim=0))
        low_weightu1=F.softmax( metau1 + meta_biasu, dim=1)
        low_weightu2=F.softmax( metau2 + meta_biasu1,dim=1)
        low_weighti1=F.softmax( metai1 + meta_biasi, dim=1)
        low_weighti2=F.softmax( metai2 + meta_biasi1,dim=1)

        # The learned matrix as the weights of the transformed network
        tembedus = (t.sum(t.multiply( (auxiembedu).unsqueeze(-1), low_weightu1), dim=1))# Equal to a two-layer linear network; Ciao and Yelp data sets are plus gelu activation function
        tembedus =  t.sum(t.multiply( (tembedus)  .unsqueeze(-1), low_weightu2), dim=1)
        tembedis = (t.sum(t.multiply( (auxiembedi).unsqueeze(-1), low_weighti1), dim=1))
        tembedis =  t.sum(t.multiply( (tembedis)  .unsqueeze(-1), low_weighti2), dim=1)
        transfuEmbed = tembedus
        transfiEmbed = tembedis
        return transfuEmbed, transfiEmbed
    def forward(self, iftraining, uid, iid, norm = 1):
        
        item_index=np.arange(0,self.itemNum)
        user_index=np.arange(0,self.userNum)
        ui_index = np.array(user_index.tolist() + [ i + self.userNum for i in item_index])
        
        # Initialize Embeddings
        userembed0 = self.embedding_dict['user_emb'].weight
        itemembed0 = self.embedding_dict['item_emb'].weight
        uu_embed0  = self.self_gatingu(userembed0)
        ii_embed0  = self.self_gatingi(itemembed0)
        self.ui_embeddings       = t.cat([ userembed0, itemembed0], 0)
        self.all_user_embeddings = [uu_embed0]
        self.all_item_embeddings = [ii_embed0]
        self.all_ui_embeddings   = [self.ui_embeddings]
        # Encoder
        for i in range(len(self.encoder)):
            layer = self.encoder[i]
            if i == 0:  
                userEmbeddings0 = layer(uu_embed0, self.uuMat, user_index)
                itemEmbeddings0 = layer(ii_embed0, self.iiMat, item_index)
                uiEmbeddings0   = layer(self.ui_embeddings, self.uiMat, ui_index)
            else:
                userEmbeddings0 = layer(userEmbeddings, self.uuMat, user_index)
                itemEmbeddings0 = layer(itemEmbeddings, self.iiMat, item_index)
                uiEmbeddings0   = layer(uiEmbeddings,   self.uiMat, ui_index)
            
            # Aggregation of message features across the two related views in the middle layer then fed into the next layer
            self.ui_userEmbedding0, self.ui_itemEmbedding0 = t.split(uiEmbeddings0, [self.userNum, self.itemNum])
            userEd=( userEmbeddings0 + self.ui_userEmbedding0 )/2.0
            itemEd=( itemEmbeddings0 + self.ui_itemEmbedding0 )/2.0
            userEmbeddings=userEd 
            itemEmbeddings=itemEd
            uiEmbeddings=torch.cat([ userEd,itemEd],0) 
            if norm == 1:
                norm_embeddings = F.normalize(userEmbeddings0, p=2, dim=1)
                self.all_user_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(itemEmbeddings0, p=2, dim=1)
                self.all_item_embeddings += [norm_embeddings]
                norm_embeddings = F.normalize(uiEmbeddings0, p=2, dim=1)
                self.all_ui_embeddings   += [norm_embeddings]
            else:
                self.all_user_embeddings += [userEmbeddings]
                self.all_item_embeddings += [norm_embeddings]
                self.all_ui_embeddings   += [norm_embeddings]
        self.userEmbedding = t.stack(self.all_user_embeddings, dim=1)
        self.userEmbedding = t.mean(self.userEmbedding, dim = 1)
        self.itemEmbedding = t.stack(self.all_item_embeddings, dim=1)  
        self.itemEmbedding = t.mean(self.itemEmbedding, dim = 1)
        self.uiEmbedding   = t.stack(self.all_ui_embeddings, dim=1)
        self.uiEmbedding   = t.mean(self.uiEmbedding, dim=1)
        self.ui_userEmbedding, self.ui_itemEmbedding = t.split(self.uiEmbedding, [self.userNum, self.itemNum])
        
        # Personalized Transformation of Auxiliary Domain Features
        metatsuembed,metatsiembed = self.metafortansform(self.userEmbedding, self.ui_userEmbedding, self.itemEmbedding, self.ui_itemEmbedding)
        self.userEmbedding = self.userEmbedding + metatsuembed
        self.itemEmbedding = self.itemEmbedding + metatsiembed
        
        # Regularization: the constraint of transformed reasonableness
        metaregloss = 0
        if iftraining == True :
            self.reg_lossu = self.metaregular((self.ui_userEmbedding[uid.cpu().numpy()]),(self.userEmbedding),self.uuMat[uid.cpu().numpy()])
            self.reg_lossi = self.metaregular((self.ui_itemEmbedding[iid.cpu().numpy()]),(self.itemEmbedding),self.iiMat[iid.cpu().numpy()])
            metaregloss =  (self.reg_lossu +  self.reg_lossi)/2.0
        return self.userEmbedding, self.itemEmbedding,(self.args.wu1*self.ui_userEmbedding + self.args.wu2*self.userEmbedding), (self.args.wi1*self.ui_itemEmbedding + self.args.wi2*self.itemEmbedding), self.ui_userEmbedding, self.ui_itemEmbedding ,  metaregloss
 
class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        import scipy.sparse as sp
        import numpy as np
        
        # 确保输入是 COO 格式的稀疏矩阵
        if not sp.isspmatrix_coo(adj):
            adj = sp.coo_matrix(adj)
        
        # 计算行和
        rowsum = np.array(adj.sum(1)).flatten()
        
        # 计算度数的逆平方根，处理零度数情况
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(rowsum, -0.5)
        
        # 将无穷大值替换为0
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        # 创建对角矩阵
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # 归一化邻接矩阵
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat)
        out_features = torch.spmm(subset_sparse_tensor, subset_features)
        new_features = torch.empty(features.shape)
        new_features[index] = out_features
        dif_index = np.setdiff1d(torch.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        return new_features

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre =   nn.Linear(input_dim, feature_dim,bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out =    nn.Linear(feature_dim, output_dim,bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu=nn.PReLU()
        x = prelu(x) 
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

class HGCL(SequentialModel):
    def __init__(self, args, corpus):
        # Initialize the parent class
        super().__init__(args, corpus)

        
        # Extract necessary information from corpus
        self.userNum = corpus.n_users
        self.itemNum = corpus.n_items
        
        # Extract adjacency matrices from corpus if available
        import scipy.sparse as sp
        userMat = corpus.social_mat if hasattr(corpus, 'social_mat') else sp.eye(self.userNum).tocsr()
        itemMat = corpus.item_mat if hasattr(corpus, 'item_mat') else sp.eye(self.itemNum).tocsr()
        uiMat = corpus.ui_mat if hasattr(corpus, 'ui_mat') else sp.eye(self.userNum + self.itemNum).tocsr()
        
        # Hyperparameters
        self.hide_dim = 64  # 强制设置为64，确保一致性
        self.LayerNums = getattr(args, 'num_layers', 2)
        self.k = getattr(args, 'rank', 2)  # Default rank if not specified
        
        # Adjacency Matrices
        self.uuMat = userMat
        self.iiMat = itemMat
        self.uiMat = uiMat
        
        # Prepare user-item adjacency matrix
        try:
            uimat = self.uiMat[:self.userNum, self.userNum:]
        except Exception:
            # Fallback method if slicing fails
            uimat = sp.csr_matrix((self.userNum, self.itemNum))
        
        values = torch.FloatTensor(uimat.data)
        indices = np.vstack((uimat.nonzero()[0], uimat.nonzero()[1]))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = torch.Size([self.userNum, self.itemNum])
        self.uiadj = torch.sparse.FloatTensor(i, v, shape)
        self.iuadj = self.uiadj.transpose(0, 1)
        
        # Gating Weights
        self.gating_weightub = nn.Parameter(torch.FloatTensor(1, self.hide_dim))
        nn.init.xavier_normal_(self.gating_weightub.data)
        self.gating_weightu = nn.Parameter(torch.FloatTensor(self.hide_dim, self.hide_dim))
        nn.init.xavier_normal_(self.gating_weightu.data)
        
        self.gating_weightib = nn.Parameter(torch.FloatTensor(1, self.hide_dim))
        nn.init.xavier_normal_(self.gating_weightib.data)
        self.gating_weighti = nn.Parameter(torch.FloatTensor(self.hide_dim, self.hide_dim))
        nn.init.xavier_normal_(self.gating_weighti.data)
        
        # Embeddings
        self.embedding_dict = nn.ModuleDict({
            'user_emb': nn.Embedding(self.userNum, self.hide_dim),
            'item_emb': nn.Embedding(self.itemNum, self.hide_dim),
            'uu_emb': nn.Embedding(self.userNum, self.hide_dim),
            'ii_emb': nn.Embedding(self.itemNum, self.hide_dim)
        })
        
        # Encoder Layers
        self.encoder = nn.ModuleList([GCN_layer() for _ in range(self.LayerNums)])
        
        # Meta Networks
        self.meta_netu = nn.Linear(self.hide_dim * 3, self.hide_dim, bias=True)
        self.meta_neti = nn.Linear(self.hide_dim * 3, self.hide_dim, bias=True)
        
        # MLPs for Personalized Transformation
        self.mlp = MLP(self.hide_dim, self.hide_dim * self.k, self.hide_dim // 2, self.hide_dim * self.k)
        self.mlp1 = MLP(self.hide_dim, self.hide_dim * self.k, self.hide_dim // 2, self.hide_dim * self.k)
        self.mlp2 = MLP(self.hide_dim, self.hide_dim * self.k, self.hide_dim // 2, self.hide_dim * self.k)
        self.mlp3 = MLP(self.hide_dim, self.hide_dim * self.k, self.hide_dim // 2, self.hide_dim * self.k)
    
        # Create an instance of the MODEL class
        self.model = MODEL(
            args=args, 
            userNum=self.userNum, 
            itemNum=self.itemNum, 
            userMat=userMat, 
            itemMat=itemMat, 
            uiMat=uiMat, 
            hide_dim=self.hide_dim, 
            Layers=self.LayerNums
        )

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        import scipy.sparse as sp
        import numpy as np
        
        # 确保输入是稀疏矩阵
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        
        # 转换为 COO 格式
        adj = adj.tocoo()
        
        # 计算行和
        rowsum = np.array(adj.sum(1)).flatten()
        
        # 处理度数的逆平方根
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(rowsum, -0.5)
        
        # 将无穷大值替换为0
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        # 创建对角矩阵
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # 归一化邻接矩阵
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def self_gatingu(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightu) + self.gating_weightub))
    
    def self_gatingi(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weighti) + self.gating_weightib))
    
    def metafortransform(self, auxiembedu, targetembedu, auxiembedi, targetembedi):
        # Neighbor information
        uneighbor = torch.matmul(self.uiadj, self.ui_itemEmbedding)
        ineighbor = torch.matmul(self.iuadj, self.ui_userEmbedding)
        
        # Meta-knowledge extraction
        tembedu = self.meta_netu(torch.cat((auxiembedu, targetembedu, uneighbor), dim=1).detach())
        tembedi = self.meta_neti(torch.cat((auxiembedi, targetembedi, ineighbor), dim=1).detach())
        
        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        metau1 = self.mlp(tembedu).reshape(-1, self.hide_dim, self.k)  # d*k
        metau2 = self.mlp1(tembedu).reshape(-1, self.k, self.hide_dim)  # k*d
        metai1 = self.mlp2(tembedi).reshape(-1, self.hide_dim, self.k)  # d*k
        metai2 = self.mlp3(tembedi).reshape(-1, self.k, self.hide_dim)  # k*d
        
        meta_biasu = torch.mean(metau1, dim=0)
        meta_biasu1 = torch.mean(metau2, dim=0)
        meta_biasi = torch.mean(metai1, dim=0)
        meta_biasi1 = torch.mean(metai2, dim=0)
        
        low_weightu1 = F.softmax(metau1 + meta_biasu, dim=1)
        low_weightu2 = F.softmax(metau2 + meta_biasu1, dim=1)
        low_weighti1 = F.softmax(metai1 + meta_biasi, dim=1)
        low_weighti2 = F.softmax(metai2 + meta_biasi1, dim=1)
        
        # The learned matrix as the weights of the transformed network
        tembedus = torch.sum(torch.multiply(auxiembedu.unsqueeze(-1), low_weightu1), dim=1)
        tembedus = torch.sum(torch.multiply(tembedus.unsqueeze(-1), low_weightu2), dim=1)
        
        tembedis = torch.sum(torch.multiply(auxiembedi.unsqueeze(-1), low_weighti1), dim=1)
        tembedis = torch.sum(torch.multiply(tembedis.unsqueeze(-1), low_weighti2), dim=1)
        
        transfuEmbed = tembedus
        transfiEmbed = tembedis
        
        return transfuEmbed, transfiEmbed
    
    def forward(self, feed_dict):
        # 提取用户和物品ID
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        
        # 如果 items 是二维的，选择第一列
        if items.ndim > 1:
            items = items[:, 0]
        
        # 调用原始模型的 forward 方法
        userEmbedding, itemEmbedding, _, _, ui_userEmbedding, ui_itemEmbedding, metaregloss = self.model.forward(
            iftraining=True, 
            uid=users, 
            iid=items
        )
        
        # 获取用户和物品嵌入
        user_emb = userEmbedding
        item_emb = itemEmbedding
        
        # 生成预测
        # 对于每个用户，计算其与所有物品的分数
        all_item_emb = self.model.embedding_dict['item_emb'].weight
        prediction_matrix = torch.matmul(user_emb, all_item_emb.t())
        
        # 尝试基于历史交互生成标签
        if 'history_items' in feed_dict and 'history_times' in feed_dict:
            history_items = feed_dict['history_items']
            history_times = feed_dict['history_times']
            
            labels = torch.tensor([
                len(hist_items) if len(hist_items) > 0 else 0 
                for hist_items in history_items
            ], dtype=torch.float)
        else:
            labels = torch.ones_like(users, dtype=torch.float)
        
        return {
            'prediction': prediction_matrix,
            'user_id': users,
            'item_id': items,
            'labels': labels,
            'metaregloss': metaregloss
        }

    def loss(self, feed_dict):
        # 获取前向传播的结果
        forward_res = self.forward(feed_dict)
        prediction = forward_res['prediction']
        users = forward_res['user_id']
        items = forward_res['item_id']
        labels = forward_res['labels']
        metaregloss = forward_res['metaregloss']
        
        # 选择预测矩阵中每个用户对目标物品的预测分数
        batch_prediction = prediction[torch.arange(prediction.size(0)), items]
        
        # 基本的预测损失（均方误差）
        base_loss = F.mse_loss(batch_prediction, labels.float())
        
        # 总损失
        meta_reg_weight = 0.1
        total_loss = base_loss + meta_reg_weight * metaregloss
        
        return total_loss

    def compute_contrastive_loss(self, user_emb, item_emb, temperature=0.5):
        """
        计算对比学习损失
        
        Args:
        - user_emb: 用户嵌入 [batch_size, embedding_dim]
        - item_emb: 物品嵌入 [batch_size, embedding_dim]
        - temperature: 温度参数，控制对比学习的软硬程度
        
        Returns:
        - contrastive_loss: 对比学习损失
        """
        # 计算用户和物品嵌入之间的相似度矩阵
        # 使用余弦相似度
        similarity_matrix = F.cosine_similarity(
            user_emb.unsqueeze(1), 
            item_emb.unsqueeze(0), 
            dim=-1
        ) / temperature
        
        # 创建正样本标签（对角线）
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        
        # 计算对比损失
        # 使用交叉熵损失，将相似度矩阵作为输入
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        return contrastive_loss


# small change
    