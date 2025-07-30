import numpy as np
import pandas as pd
import os

class HfssDataset:
    """HFSS仿真数据集管理库
    
    功能:
    1. 高效提取复数S参数实部/虚部
    2. 构建GPR训练所需的数据结构
    3. 数据集持久化
    4. 频率点管理
    
    设计原则:
    - 独立于约束计算逻辑
    - 保持特征提取的通用性
    - 优化大型数据集性能
    """
    
    def __init__(self, variables, freq_range, n_freq_points=20):
        """
        初始化数据集
        
        参数:
        variables: List[dict] - 变量定义
        freq_range: Tuple[float, float] - 频率范围(Hz)
        n_freq_points: int - 频率点数量
        """
        self.variables = variables
        self.freq_range = freq_range
        self.n_freq_points = n_freq_points
        self.feature_names = [v['name'] for v in variables]
        
        # 生成目标频率点(GHz)
        self.target_freqs = np.linspace(
            freq_range[0] / 1e9, 
            freq_range[1] / 1e9, 
            n_freq_points
        )
        
        # 修改初始化
        self.X = []  # 改为列表存储参数
        self.y = []  # 改为列表存储特征
        self.feature_vectors = []  # 存储展平的特征向量
        
        # 端口映射缓存
        self.port_mappings = {}
    
    def add_port_mapping(self, name, tx_port, rx_port):
        """添加端口映射关系"""
        self.port_mappings[name] = (tx_port, rx_port)
    
    def extract_complex_features(self, s_params_df):
        features = {}
        
        for name, (tx, rx) in self.port_mappings.items():
            s_col = f"S({tx},{rx})"
            
            if s_col not in s_params_df.columns:
                print(f"⚠️ 警告: S参数列 {s_col} 不在仿真结果中")
                continue
                
            # 创建存储数组
            s_complex = np.zeros((self.n_freq_points, 2))
            
            # 获取仿真频率(GHz)
            sim_freqs = s_params_df['Frequency']
            
            # 对每个目标频率点进行插值
            for i, target_freq in enumerate(self.target_freqs):
                # 找到最接近的频率索引
                idx = np.abs(sim_freqs - target_freq).argmin()
                
                # 获取该频率点的复数S参数
                s_val = s_params_df[s_col].iloc[idx]
                s_complex[i, 0] = np.real(s_val)  # 实部
                s_complex[i, 1] = np.imag(s_val)  # 虚部
            
            features[name] = s_complex
        
        return features
    
    def add_sample(self, sample_params, s_params_features):
        # 展平特征为向量
        flat_features = []
        for port_name in self.port_mappings:
            if port_name in s_params_features:
                flat_features.extend(s_params_features[port_name].ravel())
        
        # 添加到数据集
        self.X.append(sample_params)
        self.y.append(s_params_features)
        self.feature_vectors.append(np.array(flat_features))
    
    def add_sample(self, sample_params, s_params_features):
        # 展平特征为向量
        flat_features = []
        for port_name in self.port_mappings:
            if port_name in s_params_features:
                flat_features.extend(s_params_features[port_name].ravel())
        
        # 添加到数据集
        self.X.append(sample_params)
        self.y.append(s_params_features)
        self.feature_vectors.append(np.array(flat_features))
    
    def get_dataset(self, port_name=None):
        """
        获取完整数据集
        
        参数:
        port_name: str - 指定端口名称(默认返回第一个端口)
        
        返回:
        X: np.array - 输入特征 (n_samples, n_variables)
        y: np.array - 输出目标 (n_samples, n_freq_points, 2)
        """
        if self.X is None or self.y is None:
            return None, None
            
        return self.X, self.y
    
    def get_flat_dataset(self):
        """获取平铺数据集用于模型训练"""
        if not self.X:
            return None, None
            
        X_array = np.array(self.X)
        y_array = np.array(self.feature_vectors)
        return X_array, y_array
    
    def save_dataset(self, file_path):
        """保存数据集到文件"""
        if self.X is None or self.y is None:
            raise RuntimeError("数据集为空")
            
        dataset = {
            'variables': self.variables,
            'freq_range': self.freq_range,
            'target_freqs': self.target_freqs,
            'port_mappings': self.port_mappings,
            'X': self.X,
            'y': self.y
        }
        np.savez(file_path, **dataset)
    
    def load_dataset(self, file_path):
        """从文件加载数据集"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        data = np.load(file_path, allow_pickle=True)
        self.variables = data['variables'].tolist()
        self.freq_range = tuple(data['freq_range'])
        self.target_freqs = data['target_freqs']
        self.port_mappings = data['port_mappings'].item()
        self.X = data['X']
        self.y = data['y']
    
    def get_frequency_points(self):
        """获取目标频率点(GHz)"""
        return self.target_freqs.copy()
    
    def size(self):
        """获取数据集大小"""
        return len(self.X) if self.X is not None else 0
    
    def clear(self):
        """清空数据集"""
        self.X = None
        self.y = None