"""HFSS AI代理优化器 - 主动学习框架
基于主动学习和混合代理模型（GPR+DNN）的HFSS天线参数优化系统
目标：实现对天线参数的优化，给出能够满足约束条件的最优参数组合
_核心功能与流程_：
一. 初始数据集构建​​
​    1.​拉丁超立方采样​​：生成初始参数组合（如天线尺寸、材料属性等设定好的参数变量），确保设计空间均匀覆盖。
​    2.​HFSS仿真与特征提取​​：
        每个参数组合通过HFSS仿真获得仿真结果（如复数S参数）
        降采样提取特征向量。
        存储为数据集：(输入参数, 特征向量)。
​    3.​关键改进​​：
        特征向量能在一定程度上代表天线的电磁特性（如复数的S参数），确保模型泛化性。
        模型训练的数据集与约束函数相互独立不关联
​二. GPR代理模型训练​​
​    1.​输入​​：设计参数（如贴片长度、馈电位置）。
    2.​输出​​：多维特征向量（如S11随频率变化的复数序列）。
    3.​训练目标​​：最小化预测特征向量与真实仿真的均方误差（MSE），不涉及约束条件。
三. 主动学习循环​​
​​每一轮迭代流程如下：​​
    1.​生成候选池​​：随机生成1000组参数组合。
    2.​GPR预测与不确定性评估​​：
        预测候选点的特征向量y和预测不确定性（标准差）。
    3.​主动学习样本筛选​​：
        -​策略​​：选择​​高不确定性​​或​​高性能潜力​​样本：
​            ​高不确定性区域​​：max(σ)>σ（threshold）（探索未充分区域）。
​            ​高性能潜力区域​​：P=-Loss 最大（损失越小，-Loss越大，越接近约束条件，如S11最接近目标频段）。
            计算损失的时候完全可以先从特征向量里返还为复数s参数，然后计算式得出dB，而非从特征向量里提取dB的特征再计算损失，
                    保证模型本身学习的是本质的S参数，而非dB的特征。
        -​​筛选公式(归一化)​​：
            Score=a⋅[σ/σ(max)] + (1−a)⋅[(P-Pmin)/(Pmax-Pmin)], 
                其中 a 平衡探索与利用(通常 a=0.6), σ 为预测不确定性, 
                P性能潜力, Pmax/Pmin为当前候选池中性能潜力的最大值和最小值。
​    4.​HFSS验证与数据集更新​​：
        对筛选出的样本进行HFSS仿真，提取真实特征向量，加入数据集。
​    5.​模型重训练​​：用扩充后的数据集更新GPR模型。
​        ​终止条件​​：
            达到最大轮次（如20轮），或GPR的MSE收敛（如变化率<5%）。
​​四. 最优解提取（独立于主动学习）​​
​    1.​预筛选​​：
        -生成10,000组随机参数，用GPR预测特征向量。
        -基于约束条件（寻找损失值最低）过滤候选解，保留Top-K（如50组）。
    2.​HFSS验证​​：
        -将候选解送入HFSS仿真，剔除预测误差大的样本。
        -输出所有满足约束的解。
    3.​失败处理​​：
        -若验证后无解，将失败样本加入数据集，重新训练GPR并重复预筛选。
"""

"""HFSS AI代理优化器 - 主动学习框架"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from pyDOE import lhs
import os
import time
import json
import re
import traceback
from typing import List, Dict, Callable, Optional, Tuple
from api import HFSSController
import warnings
from DataSet import HfssDataset
from GPRtrainer import GPRTrainer

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

class HfssAIAgentOptimizer:
    """HFSS AI代理优化器（重构版）"""
    
    def __init__(self, 
                 project_path: str,
                 design_name: str = "HFSSDesign1",
                 setup_name: str = "Setup1",
                 sweep_name: str = "Sweep",
                 variables: List[dict] = None,
                 freq_range: Tuple[float, float] = (5.5e9, 7e9),
                 constraints: List[dict] = None,
                 global_port_map: Dict[str, Tuple[str, str]] = None,
                 output_dir: str = "optim_results",
                 max_active_cycles: int = 30,   # 最大主动学习轮次
                 init_sample_multiplier: int = 10, # 初始样本数量的倍数
                 feature_freq_points: int = 20, # 特征向量的频率点数量
                 n_select: int = 3, # 每轮主动学习选择的样本数
                 ei_balance: float = 0.6, # 探索与利用的平衡因子
                 initial_dataset_path: Optional[str] = None,  # 新增：初始数据集路径
                 log_level: int = 1,  # 新增日志级别参数: 0=静默,1=正常,2=详细
                 ):
        
        # HFSS配置
        self.project_path = project_path
        self.design_name = design_name
        self.setup_name = setup_name
        self.sweep_name = sweep_name
        
        # 优化参数
        self.variables = variables or []
        self.freq_range = freq_range
        self.constraints = constraints or []
        self.global_port_map = global_port_map or {}
        self.output_dir = output_dir
        self.max_active_cycles = max_active_cycles
        self.init_sample_multiplier = init_sample_multiplier
        self.feature_freq_points = min(feature_freq_points, 50)
        self.n_select = n_select
        self.ei_balance = ei_balance
        self.log_level = log_level  # 保存日志级别
        
        # 内部状态
        self.hfss = None
        self.active_cycle = 0
        self.best_loss = float('inf')
        self.port_name_map = {}
        self.start_time = None

        # 创建数据集实例
        self.dataset = HfssDataset(
            variables=self.variables,
            freq_range=self.freq_range,
            n_freq_points=self.feature_freq_points
        )
        # 添加端口映射
        for sp_name, ports in self.global_port_map.items():
            self.dataset.add_port_mapping(sp_name, ports[0], ports[1])

        self.gpr = None
        self.gpr_trainer = None  # 新增：GPR训练器实例引用
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        self.loss_history = []        # 存储每轮评估样本的最小损失
        self.mse_history = []         # 存储每轮GPR训练后的训练集MSE
        self.unique_samples = set()   # 追踪已探索的设计点

        # 新增：初始数据集路径
        self.initial_dataset_path = initial_dataset_path
        self.dataset_version = 1.0
        # 修改目录创建逻辑
        if initial_dataset_path and os.path.exists(initial_dataset_path):
            # 使用数据集所在目录作为工作目录
            self.save_dir = os.path.dirname(initial_dataset_path)
            print(f"💾 使用现有工作目录: {self.save_dir}")
        else:
            # 创建新的时间戳目录
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.save_dir = os.path.join(self.output_dir, f"ai_optim_{timestamp}")
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"💾 创建新工作目录: {self.save_dir}")

        # 生成特征列名
        self._generate_feature_columns()
        
        print(f"✅ AI优化器初始化完成 | 特征维度: {len(self.feature_columns)}"
              f"{' | 使用预加载数据集' if initial_dataset_path else ''}")

    def _generate_feature_columns(self):
        """生成标准化特征列名"""
        self.feature_columns = []
        
        # 生成目标频率点
        target_freqs = np.linspace(
            self.freq_range[0] / 1e9, 
            self.freq_range[1] / 1e9, 
            self.feature_freq_points
        )
        
        # 为每个S参数和频率点创建实部/虚部列
        for sp_name in self.global_port_map.keys():
            for freq in target_freqs:
                self.feature_columns.append(f"{sp_name}_{freq:.2f}GHz_real")
                self.feature_columns.append(f"{sp_name}_{freq:.2f}GHz_imag")

    # ======================== 核心优化流程 ========================
    def optimize(self):
        """运行AI代理优化流程"""
        print(f"\n{'='*60}")
        print("🚀 启动 HFSS AI代理优化")
        print(f"优化变量: {[v['name'] for v in self.variables]}")
        print(f"约束条件: {[c['expression'] for c in self.constraints]}")
        print(f"最大轮次: {self.max_active_cycles}")
        print('='*60)
        
        self.start_time = time.time()
        
        try:
            # 初始化HFSS环境
            self._initialize_hfss_environment()
            
            # ===== 新增：初始数据集加载逻辑 =====
            if self.initial_dataset_path:
                print("\n" + "="*60)
                print(f"🔁 加载初始数据集: {self.initial_dataset_path}")
                print('='*60)
                
                if not self.load_dataset(self.initial_dataset_path):
                    print("❌ 数据集加载失败，使用初始采样")
                    initial_samples = self.generate_initial_samples()
                    self.evaluate_samples(initial_samples, "Initial")
                    # 新数据集才需要保存
                    self.save_dataset()  
                else:
                    print(f"✅ 成功加载数据集 | 样本数: {self.dataset.size()}")

                    # 验证数据集
                    if self.verify_dataset(self.initial_dataset_path):
                        print("✅ 数据集验证通过")
                    else:
                        print("❌ 数据集验证失败，内容不一致")
                    
                    # 显示数据集摘要
                    self.show_dataset_summary()
                    
                    # 导出为CSV
                    csv_path = os.path.join(self.save_dir, "dataset_export.csv")
                    self.export_dataset_to_csv(csv_path)
                    # 使用已有数据集时不重复保存
            else:
                # 生成初始样本
                print("\n" + "="*60)
                print("🌟 开始初始样本采样")
                print('='*60)
                initial_samples = self.generate_initial_samples()
                self.evaluate_samples(initial_samples, "Initial")
                # 新数据集需要保存
                self.save_dataset() 
            
            # 训练GPR模型
            print("\n" + "="*60)
            print("🧠 训练GPR代理模型")
            print('='*60)
            # 训练GPR模型
            if not self.train_feature_model():
                print("❌ GPR模型训练失败，终止优化")
            else:
                # +++ 新增：训练后立即验证模型 +++
                self.validate_model(n_samples=5)

            # 主动学习策略
            # +++ 新增：执行主动学习循环 +++
            self.active_learning_cycle()
            
            # +++ 新增：最优解提取 +++
            self.extract_optimal_solutions()
            
        except Exception as e:
            print(f"❌ 优化失败: {str(e)}")
            traceback.print_exc()
        finally:
            # 导出为CSV
            if self.dataset:
                csv_path = os.path.join(self.save_dir, "dataset_export.csv")
                self.export_dataset_to_csv(csv_path)
            # 确保关闭HFSS连接
            if self.hfss:
                
                self.hfss.close()

    def _initialize_hfss_environment(self):
        """初始化HFSS环境"""
        self.hfss = HFSSController(
            project_path=self.project_path,
            design_name=self.design_name,
            setup_name=self.setup_name,
            sweep_name=self.sweep_name
        )
        
        if not self.hfss.connect():
            raise RuntimeError("HFSS连接失败")
        
        if not self.hfss.check_design_config():
            raise RuntimeError("设计配置检查失败")
        
        # 构建端口名称映射
        self.build_port_name_map()

    def build_port_name_map(self):
        """构建端口名称映射（修复版本）"""
        ports = self.hfss.get_ports()
        print(f"🔌🔌🔌🔌 检测到的端口: {ports}")
        
        self.port_name_map = {}
        for i, port in enumerate(ports, 1):
            # 保留原始端口名称，只移除空格（冒号是有效字符）
            clean_port = port.strip().replace(" ", "")  # 仅移除空格
            self.port_name_map[str(i)] = clean_port
        
        # 更新全局端口映射（使用原始名称）
        for sp_name, (tx_id, rx_id) in self.global_port_map.items():
            tx_port = self.port_name_map.get(tx_id, tx_id)
            rx_port = self.port_name_map.get(rx_id, rx_id)
            # 确保端口名称格式为 "端口号:端口号"
            self.global_port_map[sp_name] = (tx_port, rx_port)
        
        print(f"🔀🔀🔀🔀 更新全局端口映射: {self.global_port_map}")

    # ======================== 采样方法 ========================
    def generate_initial_samples(self) -> np.ndarray:
        """生成拉丁超立方初始样本"""
        n_vars = len(self.variables)
        n_samples = max(5, n_vars * self.init_sample_multiplier)
        
        print(f"📊 生成初始样本: {n_samples}个点")
        
        # 生成拉丁超立方样本
        lhs_samples = lhs(n_vars, samples=n_samples, criterion='maximin')
        
        # 映射到实际参数范围
        samples = np.zeros_like(lhs_samples)
        for i, var in enumerate(self.variables):
            low, high = var['bounds']
            samples[:, i] = lhs_samples[:, i] * (high - low) + low
        
        return samples

    # 添加最优解提取方法
    def extract_optimal_solutions(self, n_candidates=10000, top_k=3):
        """提取最优解（独立于主动学习）"""
        print("\n" + "="*60)
        print("🏆🏆 开始最优解提取")
        print('='*60)
        
        # 1. 预筛选
        print(f"🔍 生成{n_candidates}个候选解...")
        candidates = self.generate_candidate_samples(n_candidates)
        
        print("🧠 使用GPR预测特征...")
        y_pred, _ = self.gpr_trainer.predict(candidates)
        
        print("📊 计算候选解损失...")
        losses = np.zeros(n_candidates)
        for i in range(n_candidates):
            losses[i] = self.calculate_potential_loss(y_pred[i])
        
        # 筛选Top-K候选解
        top_indices = np.argsort(losses)[:top_k]
        top_candidates = candidates[top_indices]
        top_losses = losses[top_indices]
        
        print(f"✅ 筛选出Top-{top_k}候选解，损失范围: [{top_losses.min():.4f}, {top_losses.max():.4f}]")
        
        # 2. HFSS验证
        print("\n🔬 开始HFSS验证...")
        valid_solutions = []
        for i, candidate in enumerate(top_candidates):
            print(f"验证候选解 {i+1}/{len(top_candidates)} - 预测损失: {top_losses[i]:.4f}")
            try:
                # 评估样本
                self.evaluate_samples(np.array([candidate]), "Validation")
                
                # 获取最新添加的样本损失
                actual_loss = self.calculate_latest_loss()
                
                # 检查预测误差
                pred_error = abs(actual_loss - top_losses[i])
                if pred_error < 0.1:  # 10%误差阈值
                    valid_solutions.append({
                        "params": candidate,
                        "loss": actual_loss
                    })
                    print(f"✅ 验证通过 | 实际损失: {actual_loss:.4f} | 误差: {pred_error:.4f}")
                else:
                    print(f"⚠️ 预测误差过大 | 实际损失: {actual_loss:.4f} | 误差: {pred_error:.4f}")
                    
                # 失败处理：将样本加入数据集
                # 无论验证是否通过，样本已加入数据集
                
            except Exception as e:
                print(f"❌ 验证失败: {str(e)}")
        
        # 3. 输出结果
        print("\n" + "="*60)
        print("🎉🎉 优化结果摘要")
        print('='*60)
        if valid_solutions:
            # 按损失排序
            valid_solutions.sort(key=lambda x: x["loss"])
            best_solution = valid_solutions[0]
            print(f"🏅 找到 {len(valid_solutions)} 个有效解")
            print(f"🥇 最优解损失: {best_solution['loss']:.4f}")
            print("最优参数:")
            for i, var in enumerate(self.variables):
                print(f"  {var['name']}: {best_solution['params'][i]:.4f} {var.get('unit', '')}")
            
            # 保存结果
            result_path = os.path.join(self.save_dir, "optimal_solutions.csv")
            self.save_solutions_to_csv(valid_solutions, result_path)
            return best_solution["params"]
        else:
            print("❌ 未找到满足要求的解")
            # 失败处理：重新训练模型并重试
            print("🔄 重新训练GPR模型并重试...")
            self.train_feature_model()
            return self.extract_optimal_solutions(n_candidates//2, top_k//2)  # 减少候选规模

    def calculate_latest_loss(self):
        """计算最新样本的损失"""
        if not self.dataset.X:
            return float('inf')
        
        # 获取最新样本的特征
        latest_features = self.dataset.feature_vectors[-1]
        
        # 重构S参数特征
        s_params_features = {}
        start_idx = 0
        for sp_name in self.global_port_map:
            n_points = len(self.dataset.target_freqs)
            sp_features = latest_features[start_idx:start_idx + n_points*2]
            s_params_features[sp_name] = sp_features.reshape(n_points, 2)
            start_idx += n_points*2
        
        # 计算损失
        return self.calculate_loss_from_features(s_params_features, verbose=0)  

    def save_solutions_to_csv(self, solutions, file_path):
        """保存最优解到CSV"""
        data = []
        for i, sol in enumerate(solutions):
            row = {"loss": sol["loss"], "rank": i+1}
            for j, var in enumerate(self.variables):
                row[var["name"]] = sol["params"][j]
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"💾 最优解已保存至: {file_path}")
        
    # ======================== 代理模型 ========================
    def train_feature_model(self) -> bool:
        """训练特征模型(GPR)"""
        print("\n🧠 开始训练GPR代理模型...")
        
        # 确保数据集已保存
        dataset_path = os.path.join(self.save_dir, "dataset.npz")
        if not self.initial_dataset_path:
            self.save_dataset()
        
        try:
            # 创建并运行GPR训练器
            trainer = GPRTrainer(
                dataset_path=dataset_path,
                output_dir=self.save_dir
            )
            self.gpr_trainer = trainer  # 存储整个训练器
            self.gpr = trainer.run()
            
            # 保存训练历史
            self.save_training_history()

            return True
        except Exception as e:
            print(f"❌ GPR训练失败: {str(e)}")
            traceback.print_exc()
            return False
            
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.save_dir, "training_history.json")
        history = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": self.dataset.size(),
            "input_dim": len(self.variables),
            "output_dim": len(self.feature_columns),
            "gpr_kernel": str(self.gpr.kernel_),
            "log_likelihood": self.gpr.log_marginal_likelihood()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
        print(f"📝 训练历史已保存至: {history_path}")

    # 新增：数据集加载方法
    def load_dataset(self, file_path: str) -> bool:
        """从文件加载数据集"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ 数据集文件不存在: {file_path}")
                return False
                
            data = np.load(file_path, allow_pickle=True)
            
            # 检查数据集兼容性
            if 'version' not in data:
                print("⚠️ 数据集无版本信息，可能不兼容")
                
            # 重置数据集
            self.dataset = HfssDataset(
                variables=self.variables,
                freq_range=self.freq_range,
                n_freq_points=self.feature_freq_points
            )
            
            # 添加端口映射
            for sp_name, ports in self.global_port_map.items():
                self.dataset.add_port_mapping(sp_name, ports[0], ports[1])
            
            # 加载数据
            self.dataset.X = data['X'].tolist()
            self.dataset.y = data['y'].tolist()
            self.dataset.feature_vectors = data['feature_vectors'].tolist()
            
            # 恢复目标频率点
            if 'target_freqs' in data:
                self.dataset.target_freqs = data['target_freqs']
            
            return True
        except Exception as e:
            print(f"❌ 数据集加载失败: {str(e)}")
            traceback.print_exc()
            return False
    # ======================== 主动学习策略 ========================
    """
    主动学习循环：
    1. 生成候选池
    2. GPR预测与不确定性评估
    3. 主动学习样本筛选
    4. HFSS验证与数据集更新
    5. 模型重训练
    6. 检查终止条件
    """
    def active_learning_cycle(self):
        """执行主动学习循环"""
        print("\n" + "="*60)
        print("🔄🔄 进入主动学习循环")
        print('='*60)
        # 保存当前日志级别
        original_log_level = self.log_level

        try:
            # 主动学习阶段临时静默
            self.log_level = 0
            # 主动学习循环
            for cycle in range(1, self.max_active_cycles + 1):
                print(f"\n🔁 主动学习轮次 {cycle}/{self.max_active_cycles}")
                
                # 1. 生成候选池
                candidate_samples = self.generate_candidate_samples(n_samples=1000)
                print(f"📊 生成候选池: {len(candidate_samples)}个样本")
                
                # 2. GPR预测与不确定性评估
                print("🧠 进行GPR预测...")
                y_pred, y_std = self.gpr_trainer.predict(candidate_samples)
                print(f"🔍 预测结果: {y_pred}",f"预测不确定性: {y_std}")
                
                # 3. 主动学习样本筛选
                selected_indices = self.select_samples(candidate_samples, y_pred, y_std)
                selected_samples = candidate_samples[selected_indices]
                print(f"🔍 筛选出{len(selected_samples)}个样本进行仿真")

                self.log_level = original_log_level
                
                # 4. HFSS验证与数据集更新
                current_cycle_losses = self.evaluate_samples(selected_samples, f"Cycle_{cycle}")

                # 再次静默进行模型更新
                self.log_level = 0

                # 5. 模型重训练
                print("🔄 更新GPR模型...")
                self.train_feature_model()
                self.save_dataset()
                # 6. 检查终止条件
                # 在评估样本后记录损失
                min_loss = min(current_cycle_losses)  # 本轮评估的最小损失
                self.loss_history.append(min_loss)
                print(f"🔍 本轮评估最小损失: {min_loss}")
                # 在训练后记录MSE - 修复访问方式
                mse = self.gpr_trainer.get_train_mse()  # 现在可以正确访问
                self.mse_history.append(mse)
                
                # 检查收敛
                if self.check_convergence(cycle):
                    print("🎯 满足收敛条件，提前终止主动学习")
                    break
        finally:
            # 恢复日志级别
            self.log_level = original_log_level
        print("\n✅✅ 主动学习循环完成")

    def generate_candidate_samples(self, n_samples=1000):
        """生成候选样本点（带边界约束）"""
        n_vars = len(self.variables)
        samples = np.random.uniform(size=(n_samples, n_vars))
        
        # 映射到实际参数范围
        for i, var in enumerate(self.variables):
            low, high = var['bounds']
            samples[:, i] = samples[:, i] * (high - low) + low
        
        return samples

    def select_samples(self, candidates, y_pred, y_std):
        """根据探索-利用平衡策略筛选样本"""
        # 1. 计算每个样本的潜在性能（P）
        losses = np.array([self.calculate_potential_loss(y) for y in y_pred])
        
        # 2. 归一化性能潜力（P）和不确定性（σ）
        P = -losses  # 损失越小，性能潜力越大
        σ = np.max(y_std, axis=1)  # 取最大标准差作为不确定性度量
        
        P_norm = (P - P.min()) / (P.max() - P.min() + 1e-8)
        σ_norm = σ / (σ.max() + 1e-8)
        
        # 3. 计算综合评分
        scores = self.ei_balance * σ_norm + (1 - self.ei_balance) * P_norm
        
        # 4. 选择得分最高的样本
        top_indices = np.argsort(scores)[-self.n_select:]
        
        return top_indices

    def calculate_potential_loss(self, y_pred):
        """从预测特征向量计算潜在损失"""
        # 重构S参数特征
        s_params_features = {}
        start_idx = 0
        for sp_name in self.global_port_map:
            n_points = len(self.dataset.target_freqs)
            # 提取该S参数的特征部分
            sp_features = y_pred[start_idx:start_idx + n_points*2]
            s_params_features[sp_name] = sp_features.reshape(n_points, 2)
            start_idx += n_points*2
        
        # 计算损失（使用与真实评估相同的约束逻辑）
        return self.calculate_loss_from_features(s_params_features)

    def calculate_loss_from_features(self, s_params_features, verbose=None):
        """从特征向量计算损失（替代仿真）"""
        # 创建虚拟的DataFrame结构
        freq_points = self.dataset.target_freqs
        s_params = pd.DataFrame({'Frequency': freq_points})
        
        # 构建复数S参数 - 使用与HFSS一致的列名格式
        for sp_name, features in s_params_features.items():
            real_part = features[:, 0]
            imag_part = features[:, 1]
            s_complex = real_part + 1j * imag_part
            
            # 获取对应的端口组合
            tx, rx = self.global_port_map[sp_name]
            
            # 创建与HFSS一致的列名格式
            col_name = f"S({tx},{rx})"
            
            # 添加到DataFrame
            s_params[col_name] = s_complex
        
        # 计算损失
        return self.calculate_loss(s_params, self.constraints, verbose=verbose)

    def check_convergence(self, cycle):
        """基于实际评估结果的收敛判断"""
        if cycle < 3:  # 至少运行5轮
            return False
            
        # 1. 损失稳定性检查 (最近3轮损失变化<1%)
        if len(self.loss_history) >= 3:
            recent_losses = self.loss_history[-3:]
            loss_change = abs(max(recent_losses) - min(recent_losses)) / max(recent_losses)
            loss_stable = loss_change < 0.0001
        else:
            loss_stable = False
            
        # 2. 模型性能饱和检查 (MSE变化<2%)
        if len(self.mse_history) >= 3:
            recent_mse = self.mse_history[-3:]
            mse_change = abs(max(recent_mse) - min(recent_mse)) / min(recent_mse)
            mse_saturated = mse_change < 0.02
        else:
            mse_saturated = False
            
        # 3. 设计空间探索检查 (已探索区域比例)
        n_unique = len(self.unique_samples)
        exploration_ratio = n_unique / (cycle * self.n_select)
        well_explored = exploration_ratio > 0.8  # 80%的设计点是新的
        
        # 综合收敛条件
        convergence_reached = loss_stable and mse_saturated
        
        # 打印收敛诊断信息
        print("\n🔄 收敛诊断:")
        print(f"  - 损失稳定性: {'稳定' if loss_stable else '不稳定'} (变化: {loss_change*100:.2f}%)")
        print(f"  - 模型性能: {'饱和' if mse_saturated else '提升中'} (MSE变化: {mse_change*100:.2f}%)")
        print(f"  - 空间探索: {exploration_ratio*100:.1f}% 新设计点")
        
        return convergence_reached

    # ======================== 评估与更新 ========================
    def evaluate_samples(self, samples: np.ndarray, eval_type: str):
        """评估样本点并更新数据集
        内容包含：获取复数S参数；
                 计算损失；
                 提取特征；
                 构建数据集。
        """
        print(f"\n🔬 评估 {len(samples)} 个样本 | 阶段: {eval_type}")
            
        success_count = 0
        current_cycle_losses = []
        for i in range(samples.shape[0]):
            print(f"📊 评估样本 {i+1}/{samples.shape[0]}")
            sample = samples[i]
            try:
                # 设置变量
                for j, var in enumerate(self.variables):
                    value = sample[j]
                    self.hfss.set_variable(var['name'], value, unit=var.get('unit'))
                
                # 运行仿真
                if not self.hfss.analyze():
                    raise RuntimeError("仿真失败")
                
                # 获取S参数
                port_combos = list(set(self.global_port_map.values()))
                s_params = self.hfss.get_s_params(
                    port_combinations=port_combos,
                    data_format="complex"
                )
                
                if s_params is None:
                    raise RuntimeError("获取S参数失败")
                #print(s_params)

                loss = self.calculate_loss(s_params, constraints=self.constraints)
                current_cycle_losses.append(loss)
                print('loss:',loss)
                # 记录设计点
                sample_hash = hash(tuple(sample))
                self.unique_samples.add(sample_hash)

                # 提取特征并添加到数据集
                features = self.dataset.extract_complex_features(s_params)
                #print('features:',features)
                self.dataset.add_sample(sample, features)
                # 获取平铺数据集
                #x_dataset, y_dataset = self.dataset.get_flat_dataset()
                #print('x_dataset:',x_dataset)
                #print('y_dataset:',y_dataset)
                success_count += 1
                
            except Exception as e:
                print(f"❌ 样本评估失败: {str(e)}")
        
        print(f"✅ 成功评估: {success_count}/{len(samples)} 个样本")
        return current_cycle_losses

    def calculate_loss(self, s_params: pd.DataFrame, constraints: List[dict], verbose: int = None) -> float:
        """计算损失值，支持复杂表达式和聚合函数"""
        verbose = verbose if verbose is not None else self.log_level
        total_loss = 0.0
        
        # 创建带dB和原始复数的副本
        s_params_ext = s_params.copy()
        for column in s_params.columns:
            if column != 'Frequency' and column.startswith('S('):
                s_params_ext[f"{column}_dB"] = 20 * np.log10(np.abs(s_params[column]))
                s_params_ext[f"{column}_real"] = np.real(s_params[column])
                s_params_ext[f"{column}_imag"] = np.imag(s_params[column])
        
        if verbose >= 1:
            print("\n约束计算结果:")
            print("-"*60)
            print(f"{'约束表达式':<30} | {'目标值':<14} | {'实际值':<14} | {'损失贡献':<10}")
            print("-"*60)
        
        # 预定义支持的聚合函数
        AGG_FUNCTIONS = {
            'mean': np.mean,
            'max': np.max,
            'min': np.min,
            'sum': np.sum
        }
        
        for constraint in constraints:
            expr = constraint['expression']
            target = constraint['target']
            operator = constraint['operator']
            weight = constraint['weight']
            freq_range = constraint.get('freq_range')
            aggregate = constraint.get('aggregate', 'mean')  # 默认使用均值聚合
            
            try:
                # 1. 筛选频率范围
                if freq_range:
                    freq_min_ghz = freq_range[0] / 1e9
                    freq_max_ghz = freq_range[1] / 1e9
                    df_sub = s_params_ext[
                        (s_params_ext['Frequency'] >= freq_min_ghz) & 
                        (s_params_ext['Frequency'] <= freq_max_ghz)
                    ]
                else:
                    df_sub = s_params_ext
                    
                if len(df_sub) == 0:
                    print(f"⚠️ 警告: 约束 '{expr}' 在频率范围内无数据点")
                    loss = 10 * weight
                    total_loss += loss
                    if verbose >= 1:
                        print(f"{expr:<30} | {target:<14.4f} | {'N/A':<14} | {loss:<10.4f}")
                    continue
                    
                # 2. 解析表达式（使用聚合函数）
                agg_func = AGG_FUNCTIONS.get(aggregate)
                actual_value = self._eval_expression(df_sub, expr, agg_func)
                
                # 3. 确保actual_value是标量
                if isinstance(actual_value, np.ndarray):
                    if actual_value.size == 1:
                        actual_value = actual_value.item()
                    else:
                        # 对于数组，使用指定的聚合函数
                        actual_value = agg_func(actual_value) if agg_func else np.mean(actual_value)
                
                # 4. 计算损失
                if operator == '<':
                    violation = max(actual_value - target, 0)
                    loss = weight * (violation ** 2)  # 平方损失
                elif operator == '>':
                    violation = max(target - actual_value, 0)
                    loss = weight * (violation ** 2)
                else:  # 等式约束
                    loss = weight * abs(actual_value - target)
                    
                total_loss += loss
                
                if verbose >= 1:
                    print(f"{expr:<30} | {target:<14.4f} | {actual_value:<14.4f} | {loss:<10.4f}")
                        
            except Exception as e:
                print(f"❌❌ 约束计算失败: {expr} | 错误: {str(e)}")
                traceback.print_exc()
                loss = 10 * weight
                total_loss += loss
                if verbose >= 1:
                    print(f"{expr:<30} | {target:<14.4f} | {'ERROR':<14} | {loss:<10.4f}")
        
        if verbose >= 1:
            print("-"*60)
            print(f"{'总损失':<30} | {'':<14} | {'':<14} | {total_loss:<10.4f}")
            print("-"*60)
        
        return total_loss

    def _eval_expression(self, df: pd.DataFrame, expr: str, agg_func=None):
        """表达式求值引擎 - 修复括号处理问题"""
        # 0. 处理带括号的表达式 - 使用栈实现括号匹配
        if '(' in expr:
            stack = []
            start_index = expr.find('(')
            for i in range(len(expr)):
                if expr[i] == '(':
                    stack.append(i)
                elif expr[i] == ')':
                    if stack:
                        start = stack.pop()
                        if not stack:  # 找到最外层匹配的括号
                            inner_expr = expr[start+1:i]
                            prefix = expr[:start].strip()
                            suffix = expr[i+1:].strip()
                            
                            # 递归解析内部表达式
                            inner_value = self._eval_expression(df, inner_expr, agg_func)
                            
                            # 如果有函数名，应用函数
                            if prefix and prefix.lower() in ['db', 'real', 'imag']:
                                if prefix.lower() == 'db':
                                    inner_value = 20 * np.log10(np.abs(inner_value))
                                elif prefix.lower() == 'real':
                                    inner_value = np.real(inner_value)
                                elif prefix.lower() == 'imag':
                                    inner_value = np.imag(inner_value)
                            
                            # 递归处理后缀表达式
                            if suffix:
                                # 构造新表达式：内部结果 + 后缀
                                new_expr = f"{inner_value}{suffix}"
                                return self._eval_expression(df, new_expr, agg_func)
                            return inner_value
            # 如果所有括号都处理完，返回原始表达式
            return self._eval_expression(df, expr.replace('(', '').replace(')', ''), agg_func)

        # 1. 处理基本运算
        if '+' in expr:
            parts = expr.split('+')
            return sum(self._eval_expression(df, p, agg_func) for p in parts)
        elif '-' in expr:
            parts = expr.split('-')
            return self._eval_expression(df, parts[0], agg_func) - sum(self._eval_expression(df, p, agg_func) for p in parts[1:])
        elif '*' in expr:
            parts = expr.split('*')
            return np.prod([self._eval_expression(df, p, agg_func) for p in parts])
        elif '/' in expr:
            parts = expr.split('/')
            val = self._eval_expression(df, parts[0], agg_func)
            for p in parts[1:]:
                val /= self._eval_expression(df, p, agg_func)
            return val

        # 2. 处理S参数引用
        if expr in self.global_port_map:
            port_name = expr
            tx, rx = self.global_port_map[port_name]
            col_name = f"S({tx},{rx})"
            
            if col_name in df.columns:
                values = df[col_name].values
                return agg_func(values) if agg_func else values
            else:
                # 尝试添加后缀
                for suffix in ['', '_dB', '_real', '_imag']:
                    full_col = col_name + suffix
                    if full_col in df.columns:
                        values = df[full_col].values
                        return agg_func(values) if agg_func else values

        # 3. 直接引用列名
        if expr in df.columns:
            values = df[expr].values
            return agg_func(values) if agg_func else values

        # 4. 尝试解析为数值
        try:
            return float(expr)
        except ValueError:
            # 尝试识别S参数变体
            if expr.lower().startswith('s') and any(char.isdigit() for char in expr):
                port_name = expr.upper()
                if port_name in self.global_port_map:
                    tx, rx = self.global_port_map[port_name]
                    col_name = f"S({tx},{rx})"
                    if col_name in df.columns:
                        values = df[col_name].values
                        return agg_func(values) if agg_func else values
            
            # 最后尝试所有可能的列名变体
            possible_cols = [col for col in df.columns if expr.lower() in col.lower()]
            if possible_cols:
                values = df[possible_cols[0]].values
                return agg_func(values) if agg_func else values
            
            # 如果所有尝试都失败，抛出错误
            raise ValueError(f"无法解析表达式: {expr}")
        
    # ======================== 辅助方法 ======================== 
    def save_dataset(self, filename='dataset.npz'):
        """只在需要时保存数据集"""
        if not self.dataset.X:
            print("⚠️ 数据集为空，不保存")
            return
             
        # 使用初始数据集路径或默认文件名
        save_path = self.initial_dataset_path or os.path.join(self.save_dir, filename)
        
        try:
            np.savez(save_path,
                    X=self.dataset.X,
                    y=self.dataset.y,
                    feature_vectors=self.dataset.feature_vectors,
                    variables=self.variables,
                    port_mappings=self.global_port_map,
                    target_freqs=self.dataset.target_freqs,
                    version=self.dataset_version)
            print(f"💾 数据集已保存至: {save_path}")
            return True
        except Exception as e:
            print(f"❌ 数据集保存失败: {str(e)}")
            return False

    def verify_dataset(self, file_path: str) -> bool:
        """验证数据集内容是否一致"""
        if not os.path.exists(file_path):
            print(f"❌ 数据集文件不存在: {file_path}")
            return False
        
        try:
            # 加载待验证的数据集
            data = np.load(file_path, allow_pickle=True)
            print(f"\n🔍 正在验证数据集: {file_path}")
            
            # 检查基本属性
            print(f"数据集版本: {data.get('version', '未知')}")
            print(f"变量数量: {len(data['variables'])}")
            print(f"样本数量: {len(data['X'])}")
            
            # 检查变量一致性
            current_vars = sorted([v['name'] for v in self.variables])
            loaded_vars = sorted([v['name'] for v in data['variables']])
            if current_vars != loaded_vars:
                print(f"❌ 变量不匹配 | 当前: {current_vars} | 加载: {loaded_vars}")
                return False
            else:
                print(f"✅ 变量匹配: {current_vars}")
            
            # 检查端口映射
            current_ports = sorted(self.global_port_map.keys())
            loaded_ports = sorted(data['port_mappings'].item().keys())
            if current_ports != loaded_ports:
                print(f"❌ 端口映射不匹配 | 当前: {current_ports} | 加载: {loaded_ports}")
                return False
            else:
                print(f"✅ 端口映射匹配: {current_ports}")
                
            # 检查频率范围
            current_freqs = self.dataset.target_freqs
            loaded_freqs = data['target_freqs']
            if not np.allclose(current_freqs, loaded_freqs):
                print(f"❌ 频率点不匹配 | 当前: {current_freqs[:5]}... | 加载: {loaded_freqs[:5]}...")
                return False
            else:
                print(f"✅ 频率点匹配")
            
            # 检查样本数据完整性
            for i in range(min(3, len(data['X']))):  # 检查前3个样本
                print(f"\n样本 #{i+1} 验证:")
                # 检查输入参数
                if not np.allclose(data['X'][i], self.dataset.X[i]):
                    print(f"❌ 输入参数不匹配")
                    return False
                else:
                    print(f"✅ 输入参数匹配")
                    
                # 检查特征向量
                if not np.allclose(data['feature_vectors'][i], self.dataset.feature_vectors[i]):
                    print(f"❌ 特征向量不匹配")
                    return False
                else:
                    print(f"✅ 特征向量匹配")
                    
            return True
            
        except Exception as e:
            print(f"❌ 数据集验证失败: {str(e)}")
            traceback.print_exc()
            return False

    def show_dataset_summary(self):
        """显示数据集摘要信息"""
        if not self.dataset.X:
            print("⚠️ 数据集为空")
            return
        
        print("\n📊 数据集摘要:")
        print("-"*60)
        print(f"样本数量: {len(self.dataset.X)}")
        print(f"输入维度: {len(self.variables)}")
        print(f"特征维度: {len(self.dataset.feature_vectors[0])}")
        
        # 显示变量范围
        print("\n变量范围:")
        for i, var in enumerate(self.variables):
            values = [sample[i] for sample in self.dataset.X]
            print(f"{var['name']}: min={min(values):.4f}, max={max(values):.4f}")
        
        # 显示特征统计
        features = np.array(self.dataset.feature_vectors)
        real_parts = features[:, ::2]  # 所有实部
        imag_parts = features[:, 1::2]  # 所有虚部
        
        print("\n特征统计:")
        print(f"实部均值: {np.mean(real_parts):.4f} ± {np.std(real_parts):.4f}")
        print(f"虚部均值: {np.mean(imag_parts):.4f} ± {np.std(imag_parts):.4f}")
        print(f"特征值范围: [{np.min(features):.4f}, {np.max(features):.4f}]")
        
        # 显示前3个样本
        print("\n前3个样本示例:")
        for i in range(min(3, len(self.dataset.X))):
            print(f"\n样本 #{i+1}:")
            # 输入参数
            params = [f"{var['name']}={val:.4f}" 
                    for var, val in zip(self.variables, self.dataset.X[i])]
            print(f"参数: {', '.join(params)}")
            
            # 特征向量(简化显示)
            features = self.dataset.feature_vectors[i]
            print(f"特征: [实部: {features[0]:.4f}, 虚部: {features[1]:.4f}, ...] "
                f"(共{len(features)}个值)")
        
        print("-"*60)

    def export_dataset_to_csv(self, file_path: str):
        """将数据集导出为CSV文件"""
        try:
            if not self.dataset.X:
                print("⚠️ 数据集为空，无法导出")
                return False
                
            # 创建DataFrame
            data = []
            for i, (x, features) in enumerate(zip(self.dataset.X, self.dataset.feature_vectors)):
                row = {f"param_{j}": val for j, val in enumerate(x)}
                row.update({f"feat_{j}": val for j, val in enumerate(features)})
                data.append(row)
                
            df = pd.DataFrame(data)
            
            # 添加列名映射
            param_names = [var['name'] for var in self.variables]
            for i, name in enumerate(param_names):
                df = df.rename(columns={f"param_{i}": name})
                
            # 导出CSV
            df.to_csv(file_path, index=False)
            print(f"💾 数据集已导出为CSV: {file_path}")
            return True
        except Exception as e:
            print(f"❌ 数据集导出失败: {str(e)}")
            return False

    def complex_to_dB(self, complex_values):
        """将复数S参数转换为dB格式"""
        return 20 * np.log10(np.abs(complex_values))
    
    def validate_model(self, n_samples=10):
        """验证模型性能：对比预测结果与真实值"""
        if self.log_level < 1:
            return
            
        print("\n" + "="*60)
        print("🧪🧪 开始模型验证")
        print('='*60)
        
        # 1. 选择验证样本
        dataset_size = self.dataset.size()
        if dataset_size == 0:
            print("⚠️ 数据集为空，无法验证")
            return
            
        # 随机选择样本索引
        sample_indices = np.random.choice(dataset_size, min(n_samples, dataset_size), replace=False)
        
        # 2. 准备数据容器
        all_results = []
        
        # 3. 对每个样本进行验证
        for idx in sample_indices:
            sample_params = np.array(self.dataset.X[idx]).reshape(1, -1)
            true_features = self.dataset.feature_vectors[idx]
            
            # 模型预测
            pred_features, _ = self.gpr_trainer.predict(sample_params)
            pred_features = pred_features[0]  # 去掉batch维度
            
            # 重构S参数
            true_s_params = self.reconstruct_s_params(true_features)
            pred_s_params = self.reconstruct_s_params(pred_features)
            
            # 转换为dB格式
            true_s_params_dB = {k: self.complex_to_dB(v) for k, v in true_s_params.items()}
            pred_s_params_dB = {k: self.complex_to_dB(v) for k, v in pred_s_params.items()}
            
            # 4. 存储结果
            sample_results = {
                'index': idx,
                'params': sample_params[0].tolist(),
                'true': true_s_params_dB,
                'pred': pred_s_params_dB
            }
            all_results.append(sample_results)
            
            # 5. 可视化对比
            self.plot_comparison(idx, true_s_params_dB, pred_s_params_dB)
        
        # 6. 保存CSV结果
        self.save_validation_csv(all_results)
        print("✅✅ 模型验证完成")
    
    def reconstruct_s_params(self, feature_vector):
        """从特征向量重构S参数"""
        s_params = {}
        start_idx = 0
        for sp_name in self.global_port_map:
            n_points = len(self.dataset.target_freqs)
            # 提取该S参数的特征部分并转换为NumPy数组
            sp_features = np.array(feature_vector[start_idx:start_idx + n_points*2])
            
            # 使用NumPy正确创建复数数组
            real_arr = sp_features[::2]  # 所有实部
            imag_arr = sp_features[1::2]  # 所有虚部
            s_complex = real_arr.astype(complex)  # 转换为复数数组
            s_complex.imag = imag_arr  # 设置虚部
            
            s_params[sp_name] = s_complex
            start_idx += n_points*2
        return s_params
    
    def plot_comparison(self, idx, true_dB, pred_dB):
        """绘制预测与真实值的对比图"""
        n_ports = len(true_dB)
        fig, axs = plt.subplots(n_ports, 1, figsize=(10, 4*n_ports))
        fig.suptitle(f"样本 #{idx} 预测与真实值对比", fontsize=16)
        
        # 获取目标频率点
        freqs = self.dataset.target_freqs
        
        for i, (sp_name, true_values) in enumerate(true_dB.items()):
            ax = axs[i] if n_ports > 1 else axs
            pred_values = pred_dB[sp_name]
            
            # 绘制曲线
            ax.plot(freqs, true_values, 'b-', label='真实值', linewidth=2)
            ax.plot(freqs, pred_values, 'r--', label='预测值', linewidth=1.5)
            
            # 计算误差
            errors = np.abs(true_values - pred_values)
            max_error = np.max(errors)
            avg_error = np.mean(errors)
            
            # 添加标注
            ax.set_title(f"{sp_name} | 最大误差: {max_error:.2f}dB, 平均误差: {avg_error:.2f}dB")
            ax.set_xlabel('频率 (GHz)')
            ax.set_ylabel('幅度 (dB)')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.save_dir, f"validation_sample_{idx}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 验证对比图已保存: {plot_path}")
    
    def save_validation_csv(self, all_results):
        """将验证结果导出为CSV"""
        csv_data = []
        
        # 生成频率列名
        freqs = [f"{freq:.2f}GHz" for freq in self.dataset.target_freqs]
        
        for result in all_results:
            for sp_name, true_values in result['true'].items():
                pred_values = result['pred'][sp_name]
                
                for freq, t_val, p_val in zip(freqs, true_values, pred_values):
                    row = {
                        'sample_index': result['index'],
                        'sp_param': sp_name,
                        'frequency': freq,
                        'true_dB': t_val,
                        'pred_dB': p_val,
                        'error': abs(t_val - p_val)
                    }
                    # 添加参数值
                    for i, var in enumerate(self.variables):
                        row[var['name']] = result['params'][i]
                    
                    csv_data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.save_dir, "model_validation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"💾 验证结果CSV已保存: {csv_path}")

# 主函数示例
def main():
    """主函数 - 优化示例"""
    # 项目配置
    PROJECT_PATH = r"C:\Users\Administrator\Desktop\huaSheng\6G\6G.aedt"
    DESIGN_NAME = "HFSSDesign5"
    SETUP_NAME = "Setup1"
    SWEEP_NAME = "Sweep"
    
    # 全局端口映射
    GLOBAL_PORT_MAP = {
        'S11': ('1:1', '1:1'),
        }
    
    # 约束配置
    CONSTRAINTS = [
        {
            'expression': 'mean(dB(S11))',  # 均方误差更平滑
            'target': -15,  # 比目标值低3dB的裕量
            'operator': '<', 
            'weight': 0.4,
            'freq_range': (5.9e9, 7.2e9),
            'aggregate': 'mean'
        },
        {
            'expression': 'dB(S11)',
            'target': -12,
            'operator': '<',  # 所有端口的最大反射系数小于-10 dB
            'weight': 0.6,
            'freq_range': (5.9e9, 6.5e9),
            'aggregate': 'max'
        },
        {
            'expression': 'dB(S11)',
            'target': -12,
            'operator': '<',  # 所有端口的最大反射系数小于-10 dB
            'weight': 0.6,
            'freq_range': (6.5e9, 7.2e9),
            'aggregate': 'max'
        },
    ]
    
    # 变量配置
    VARIABLES = [
        {'name': 'Lp', 'bounds': (3, 30), 'unit': 'mm'},
        {'name': 'Lp_extra', 'bounds': (2, 20), 'unit': 'mm'},
        {'name': 'Wg', 'bounds': (1, 2.5), 'unit': 'mm'},
        {'name': 'Wp', 'bounds': (3, 25), 'unit': 'mm'},
        {'name': 'kLc', 'bounds': (0.2, 0.9), 'unit': 'meter'}, 
        {'name': 'kLm', 'bounds': (0.2, 0.8), 'unit': 'meter'},
        {'name': 'kWc', 'bounds': (0.2, 0.9), 'unit': 'meter'},
        {'name': 'kWm', 'bounds': (0.2, 1), 'unit': 'meter'},
    ]
    
    # 创建优化器实例
    optimizer = HfssAIAgentOptimizer(
        project_path=PROJECT_PATH,
        design_name=DESIGN_NAME,
        setup_name=SETUP_NAME,
        sweep_name=SWEEP_NAME,
        variables=VARIABLES,
        freq_range=(4.5e9, 8e9),
        constraints=CONSTRAINTS,
        global_port_map=GLOBAL_PORT_MAP,
        max_active_cycles=20,
        init_sample_multiplier=15,
        ei_balance=0.6,
        feature_freq_points=70,
        initial_dataset_path=r'optim_results\ai_optim_20250729-182300\dataset.npz',
    )
    
    # 开始优化
    best_params = optimizer.optimize()
    if best_params is not None:
        print("\n" + "="*60)
        print("🎉🎉 优化成功完成！最佳参数:")
        for i, var in enumerate(optimizer.variables):
            print(f"  {var['name']}: {best_params[i]:.4f} {var.get('unit', '')}")
        print('='*60)
    else:
        print("\n❌ 优化未找到有效解")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 程序异常: {str(e)}")
        traceback.print_exc()