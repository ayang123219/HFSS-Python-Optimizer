from ast import mod
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

class GPRTrainer:
    """高斯过程回归(GPR)模型训练器"""
    
    def __init__(self, dataset_path, output_dir="gpr_models"):
        """
        初始化GPR训练器
        
        参数:
        dataset_path: str - 数据集文件路径(.npz)
        output_dir: str - 模型保存目录
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.gpr = None
        self.X_train = None
        self.y_train = None  # 新增属性
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_dataset(self):
        """加载数据集"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        data = np.load(self.dataset_path, allow_pickle=True)
        return data
    
    def preprocess_data(self, data):
        """数据预处理"""
        # 提取数据和特征向量
        X = data['X']
        y = data['feature_vectors']  # 使用展平的特征向量
        
        # 标准化数据
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        
        return X_scaled, y_scaled
    
    def create_gpr_model(self):
        """创建GPR模型"""
        # 定义复合核函数
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        
        # 创建GPR模型
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,  # 优化重启次数，提高收敛概率
            alpha=1e-6,               # 增加数值稳定性
            normalize_y=True          # 自动归一化目标值
        )
    
    def train(self):
        """训练GPR模型"""
        # 加载数据集
        data = self.load_dataset()
        # 预处理数据
        X, y = self.preprocess_data(data)

        # 保存标准化后的训练数据
        self.X_train = X
        self.y_train = y

        # 创建并训练模型
        self.gpr = self.create_gpr_model()
        self.gpr.fit(X, y)
        
        # 打印训练结果
        print(f"✅ GPR模型训练完成")
        print(f"最终核函数: {self.gpr.kernel_}")
        print(f"对数边缘似然: {self.gpr.log_marginal_likelihood():.2f}")
        
        return self.gpr

    def predict(self, X):
        """
        使用训练好的GPR模型进行预测
        
        参数:
        X: np.array - 输入参数 (n_samples, n_features)
        
        返回:
        y_pred: np.array - 预测的特征向量
        y_std: np.array - 预测的标准差（不确定性）
        """
        if self.gpr is None:
            raise RuntimeError("模型未训练，请先调用train()方法")
        
        # 标准化输入
        X_scaled = self.X_scaler.transform(X)
        
        # 预测
        y_pred_scaled, y_std_scaled = self.gpr.predict(X_scaled, return_std=True)
        
        # 反标准化输出
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        y_std = y_std_scaled * self.y_scaler.scale_  # 调整标准差
        
        return y_pred, y_std

    def evaluate(self, X_test=None, y_test=None):
        """评估模型性能"""
        if self.gpr is None:
            raise RuntimeError("请先训练模型")
        
        data = self.load_dataset()
        X, y = self.preprocess_data(data)
        
        # 使用全部数据评估
        y_pred, y_std = self.gpr.predict(X, return_std=True)
        
        # 计算MSE
        mse = np.mean((y_pred - y)**2)
        print(f"平均MSE: {mse:.4f}")
        
        # 可视化预测不确定性
        self.plot_uncertainty(y, y_pred, y_std)
        
        return mse
    
    def plot_uncertainty(self, y_true, y_pred, y_std):
        """可视化预测不确定性 - 修复版"""
        plt.figure(figsize=(12, 6))
        n_features = y_true.shape[1]
        sample_dim = min(5, n_features)
        dims = np.random.choice(n_features, sample_dim, replace=False)
        
        for i, dim in enumerate(dims):
            plt.subplot(2, 3, i+1)
            plt.scatter(range(len(y_true)), y_true[:, dim], c='k', s=5, label="真实值")
            plt.plot(y_pred[:, dim], 'r-', label="预测值")
            
            # 修复：使用该维度的标准差
            plt.fill_between(
                range(len(y_pred)),
                y_pred[:, dim] - 1.96 * y_std[:, dim],
                y_pred[:, dim] + 1.96 * y_std[:, dim],
                alpha=0.2,
                color='blue',
                label="95%置信区间"
            )
            plt.title(f"特征维度 {dim}")
            plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "gpr_uncertainty.png")
        plt.savefig(plot_path)
        print(f"📊📊 不确定性可视化已保存至: {plot_path}")

    def get_train_mse(self):
        """获取模型在训练集上的MSE"""
        if self.gpr is None:
            return float('inf')
            
        # 使用训练数据计算MSE
        y_train_pred = self.gpr.predict(self.X_train)
        mse = np.mean((y_train_pred - self.y_train) ** 2)
        return mse

    def save_model(self):
        """保存训练好的模型"""
        if self.gpr is None:
            raise RuntimeError("没有可保存的模型")
        
        model_path = os.path.join(self.output_dir, "gpr_model.npz")
        params = {
            'kernel': self.gpr.kernel_,
            'X_train': self.gpr.X_train_,
            'y_train': self.gpr.y_train_
        }
        np.savez(model_path, **params)
        print(f"💾 GPR模型已保存至: {model_path}")
        
        # 保存标准化器
        scaler_path = os.path.join(self.output_dir, "scalers.npz")
        np.savez(scaler_path, 
                 X_scaler_mean=self.X_scaler.mean_,
                 X_scaler_scale=self.X_scaler.scale_,
                 y_scaler_mean=self.y_scaler.mean_,
                 y_scaler_scale=self.y_scaler.scale_)
        print(f"💾 标准化器已保存至: {scaler_path}")

    def run(self):
        """完整训练流程"""
        self.train()
        self.evaluate()
        self.save_model()
        return self.gpr



import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd 
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import json
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dropout

class BNNTrainer:
    """贝叶斯神经网络(BNN)训练器 - 重构版"""
    
    def __init__(self, dataset_path, output_dir="bnn_models", 
                 n_units=20, n_samples=15, learning_rate=0.0001):
        """
        初始化BNN训练器
        
        参数:
        dataset_path: str - 数据集文件路径(.npz)
        output_dir: str - 模型保存目录
        n_units: int - 隐藏层神经元数量
        n_samples: int - 预测时的采样次数
        learning_rate: float - 学习率
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.n_units = n_units
        self.n_samples = n_samples
        self.lr = learning_rate
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = None
        self.train_mse = float('inf')  # 存储训练集MSE
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _build_bnn(self, input_dim, output_dim):
        """构建贝叶斯神经网络结构（修复版本）"""
        inputs = Input(shape=(input_dim,))
        
        # 使用默认的均值场后验分布
        kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn()
        
        # 第一贝叶斯层
        x = tfp.layers.DenseFlipout(
            self.n_units, 
            activation='relu',
            kernel_posterior_fn=kernel_posterior_fn
        )(inputs)
        
        # 第二贝叶斯层
        x = tfp.layers.DenseFlipout(
            self.n_units, 
            activation='relu',
            kernel_posterior_fn=kernel_posterior_fn
        )(x)

        # 添加Dropout层
        x = Dropout(0.3)(x)
    
        # 输出层
        outputs = tfp.layers.DenseFlipout(
            output_dim,
            kernel_posterior_fn=kernel_posterior_fn,
            activation='softplus'
        )(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def _negative_log_likelihood(self, y_true, y_pred):
        """自定义损失函数：负对数似然"""
        # 使用MSE作为损失函数
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def load_and_preprocess(self):
        data = np.load(self.dataset_path, allow_pickle=True)
        
        # 验证数据集结构
        required_fields = ['X', 'feature_vectors']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"数据集缺少必要字段: {field}")
        
        X = data['X']
        y = data['feature_vectors']
        
        print(f"数据集验证通过: 输入维度={X.shape}, 输出维度={y.shape}")
        
        # 标准化数据
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        
        return X_scaled, y_scaled
    
    def train(self, epochs=800, batch_size=32):
        X, y = self.load_and_preprocess()
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        
        self.model = self._build_bnn(input_dim, output_dim)
        
        # 使用内置MSE损失替代自定义损失
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), 
                           loss='mse')
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,  # 显示训练进度
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15,monitor='val_loss', restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6)
            ],
            validation_split=0.2
        )
        
        self.train_mse = history.history['loss'][-1]
        print(f"✅ BNN训练完成 | 最终训练损失: {self.train_mse:.4f}")
        self.save_training_history(history)
        return history
    
    def _calculate_train_mse(self, X, y):
        """计算训练集上的MSE"""
        if self.model is None:
            return float('inf')
            
        # 使用模型预测训练集
        y_pred = self.model(X)  # 直接获取预测值
        
        # 计算MSE - 使用TensorFlow的MSE函数
        mse = tf.keras.losses.MSE(y, y_pred)
        return tf.reduce_mean(mse).numpy()
    
    def get_train_mse(self):
        """获取训练集MSE（用于主动学习收敛判断）"""
        return self.train_mse
    
    def predict(self, X):
        """预测特征向量及不确定性"""
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用train()方法")
        
        # 检查输入维度
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # 标准化输入
        X_scaled = self.X_scaler.transform(X)
        
        # 多次采样预测
        y_pred_samples = []
        for _ in range(self.n_samples):
            y_pred = self.model(X_scaled).numpy()
            y_pred_samples.append(y_pred)
        
        # 计算均值和标准差
        y_pred_samples = np.array(y_pred_samples)
        y_pred_mean = np.mean(y_pred_samples, axis=0)
        y_pred_std = np.std(y_pred_samples, axis=0)
        
        # 反标准化
        y_pred = self.y_scaler.inverse_transform(y_pred_mean)
        y_std = y_pred_std * self.y_scaler.scale_
        
        return y_pred, y_std
    
    def save_training_history(self, history):
        """保存BNN训练历史"""
        if history is None:
            print("⚠️ 无训练历史可保存")
            return
            
        # 创建历史数据
        training_history = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": list(range(1, len(history.history['loss']) + 1)),
            "loss": history.history['loss'],
            "val_loss": history.history.get('val_loss', [])
        }
        
        # 保存到文件
        history_path = os.path.join(self.output_dir, "bnn_training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"📝 BNN训练历史已保存至: {history_path}")
        return training_history
        
    def get_model_summary(self):
        if self.model is None:
            return {}
        
        return {
            "model_type": "BNN",
            "input_dim": int(self.model.input_shape[1]),  # 显式转换
            "output_dim": int(self.model.output_shape[1]),  # 显式转换
            "params_count": int(self.model.count_params()),  # 显式转换
            "train_mse": float(self.train_mse)  # 显式转换
        }

    def save_model(self):
        """保存模型及预处理器"""
        if self.model is None:
            raise RuntimeError("没有可保存的模型")
        
        # 保存模型
        model_path = os.path.join(self.output_dir, "bnn_model.h5")
        self.model.save(model_path)
        
        # 保存标准化器
        scaler_path = os.path.join(self.output_dir, "scalers.npz")
        np.savez(scaler_path,
                 X_scaler_mean=self.X_scaler.mean_,
                 X_scaler_scale=self.X_scaler.scale_,
                 y_scaler_mean=self.y_scaler.mean_,
                 y_scaler_scale=self.y_scaler.scale_)
        
        print(f"💾 BNN模型保存至: {model_path}")
        print(f"💾 标准化器保存至: {scaler_path}")
    
    def run(self):
        """完整训练流程"""
        history = self.train()
        self.save_model()
        return self.model