"""HFSS 自定义约束优化器 (增强版)
支持任意S参数组合和数学函数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real
from skopt.utils import use_named_args
import os
import time
import json
import re
import traceback
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Callable, Optional, Union
from api import HFSSController
import threading

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 
    'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HfssAdvancedConstraintOptimizer:
    """HFSS 高级约束优化器
    
    设计特点：
    - 支持任意S参数组合 (复数形式或dB形式)
    - 支持数学函数 (max, min, abs, log10, etc.)
    - 支持在表达式中使用 dB(Sxx) 语法
    - 支持复数操作 (幅值、相位、实部、虚部)
    """

    def __init__(self, 
                 project_path: str,
                 design_name: str = "HFSSDesign1",
                 setup_name: str = "Setup1",
                 sweep_name: str = "Sweep",
                 variables: List[dict] = None,
                 freq_range: Tuple[float, float] = (5.5e9, 7e9),
                 constraints: List[dict] = None,
                 global_port_map: Dict[str, Tuple[str, str]] = None,
                 max_iter: int = 30,
                 output_dir: str = "optim_results",
                 iteration_timeout: float = 1200, # 迭代超时时间 (秒)
                 max_retries: int = 3): # 最大重试次数
        
        # 添加全局后端设置
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt  # 必须在设置后端后导入

        """
        初始化优化器
        
        :param project_path: HFSS 项目路径
        :param design_name: 设计名称
        :param setup_name: 仿真设置名称
        :param sweep_name: 扫频名称
        :param variables: 优化变量列表
        :param freq_range: 基础频率范围 (Hz)
        :param constraints: 约束条件列表
        :param global_port_map: S参数名称到端口对的映射
        :param max_iter: 最大优化迭代次数
        :param output_dir: 输出目录
        """
        # HFSS 配置
        self.project_path = project_path
        self.design_name = design_name
        self.setup_name = setup_name
        self.sweep_name = sweep_name
        
        # 优化参数
        self.variables = variables or []
        self.freq_range = freq_range
        self.constraints = constraints or []
        self.global_port_map = global_port_map or {}
        self.max_iter = max_iter
        self.output_dir = output_dir
        # 添加超时相关参数
        self.iteration_timeout = iteration_timeout
        self.max_retries = max_retries
        self.timeout_count = 0  # 记录超时发生次数
        
        # 内部状态
        self.hfss = None
        self.iteration = 0
        self.history = []
        self.best_loss = float('inf')
        self.best_params = None
        self.best_result = None
        self.start_time = None
        self.port_name_map = {}

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 验证约束配置
        self.validate_constraints()\
            
        print(f"✅ 优化器初始化完成 | 约束数量: {len(self.constraints)} | 频率范围: {self.freq_range[0]/1e9}-{self.freq_range[1]/1e9} GHz")
    
    def validate_constraints(self):
        """验证约束配置有效性"""
        required_keys = ['expression', 'target', 'operator']
        for i, constraint in enumerate(self.constraints):
            # 检查必需字段
            for key in required_keys:
                if key not in constraint:
                    raise ValueError(f"约束 #{i+1} 缺少必需字段: {key}")
            
            # 检查运算符有效性
            if constraint['operator'] not in [">", ">=", "<", "<=", "=="]:
                raise ValueError(f"约束 #{i+1} 无效的运算符: {constraint['operator']}")
            
            # 检查频率设置
            if 'freq_range' in constraint and 'freq_point' in constraint:
                raise ValueError(f"约束 #{i+1} 不能同时设置 freq_range 和 freq_point")
            
            # 设置默认值
            constraint.setdefault('weight', 1.0)
            constraint.setdefault('aggregate', 'mean')
        
        # 验证全局端口映射
        for expr in [c['expression'] for c in self.constraints]:
            # 提取表达式中的所有S参数标识符
            sparams = set(re.findall(r'[a-zA-Z0-9_]+\(S\d+\)|S\d+', expr))
            for sp in sparams:
                # 处理带函数的S参数 (如 dB(S11))
                if '(' in sp and ')' in sp:
                    sp_name = sp.split('(')[1].split(')')[0]
                else:
                    sp_name = sp
                
                if sp_name not in self.global_port_map:
                    raise ValueError(f"S参数 {sp_name} 未在全局端口映射中定义")
        
        print(f"✅ 约束配置验证通过: {[c['expression'] for c in self.constraints]}")
    
    def update_frequency_range(self):
        """更新仿真频率范围到目标频段"""
        try:
            if not self.hfss or not self.hfss.hfss:
                raise RuntimeError("未连接到 HFSS")
            
            print(f"🔄🔄🔄🔄 更新仿真频率范围: {self.freq_range[0]/1e9}-{self.freq_range[1]/1e9} GHz")
            
            # 获取当前 Setup
            setup = self.hfss.hfss.get_setup(self.setup_name)
            if not setup:
                raise ValueError(f"无法获取 Setup 对象: {self.setup_name}")
            
            # 创建或更新扫频设置
            if self.sweep_name:
                # 更新现有扫频
                setup.props["FrequencySweepSetupData"] = {
                    "Type": "LinearStep",
                    "RangeType": "LinearStep",
                    "RangeStart": f"{self.freq_range[0]}Hz",
                    "RangeEnd": f"{self.freq_range[1]}Hz",
                    "RangeStep": f"{int((self.freq_range[1]-self.freq_range[0])/100)}Hz",
                }
                setup.update()
                print(f"✅ 更新扫频 '{self.sweep_name}' 成功")
            else:
                # 创建新扫频
                self.sweep_name = "OptimSweep"
                self.hfss.hfss.create_frequency_sweep(
                    setupname=self.setup_name,
                    sweepname=self.sweep_name,
                    freq_start=self.freq_range[0],
                    freq_stop=self.freq_range[1],
                    num_of_freq_points=101,
                    sweep_type="Interpolating"
                )
                print(f"✅ 创建新扫频 '{self.sweep_name}' 成功")
            
            return True
        except Exception as e:
            print(f"❌❌❌❌ 更新频率范围失败: {str(e)}")
            return False
    
    def create_objective_function(self) -> Callable:
        """创建优化目标函数"""
        
        # 定义搜索空间
        dimensions = []
        for var in self.variables:
            dimensions.append(Real(name=var['name'], low=var['bounds'][0], high=var['bounds'][1]))

        @use_named_args(dimensions=dimensions)
        def objective(**params) -> float:
            """优化目标函数"""
            self.iteration += 1
            iter_start = time.time()
            print(f"\n{'='*60}")
            print(f"🚀🚀🚀🚀 开始优化迭代 #{self.iteration}/{self.max_iter}")
            print(f"⏱⏱⏱ 超时设置: {self.iteration_timeout}秒")

            max_retries = self.max_retries  # 最大重试次数
            retry_count = 0
            loss = 1000 + self.iteration  # 默认损失值（失败时返回）
            
            while retry_count < max_retries:
                try:
                    
                    # 设置变量
                    for name, value in params.items():
                        # 查找变量单位
                        var_info = next((v for v in self.variables if v['name'] == name), None)
                        unit = var_info['unit'] if var_info else None
                        self.hfss.set_variable(name, value, unit=unit)
                    
                    # 运行仿真（带超时监控）
                    if not self.run_simulation_with_timeout():
                        raise RuntimeError("仿真失败")

                    
                    # 获取所有需要的S参数
                    all_ports = set()
                    for ports in self.global_port_map.values():
                        all_ports.add(ports)
                    
                    # 获取S参数数据 (同时获取复数和dB格式)
                    s_params = self.hfss.get_s_params(
                        port_combinations=list(all_ports),
                        data_format="both"
                    )
                    
                    if s_params is None:
                        raise RuntimeError("获取S参数失败")
                    
                    # 筛选目标频段 (单位: GHz)
                    freq_min_ghz = self.freq_range[0] / 1e9
                    freq_max_ghz = self.freq_range[1] / 1e9
                    freq_mask = (s_params['Frequency'] >= freq_min_ghz) & (s_params['Frequency'] <= freq_max_ghz)
                    s_params_band = s_params[freq_mask]
                    
                    if len(s_params_band) == 0:
                        min_freq = s_params['Frequency'].min()
                        max_freq = s_params['Frequency'].max()
                        raise RuntimeError(
                            f"目标频段内无数据\n"
                            f"  仿真频率范围: {min_freq}-{max_freq} GHz\n"
                            f"  目标频率范围: {freq_min_ghz}-{freq_max_ghz} GHz"
                        )
                    
                    # 计算损失函数
                    loss = self.calculate_constraint_loss(s_params_band)
                    
                    # 记录迭代结果
                    iter_time = time.time() - iter_start
                    iter_data = {
                        'iteration': self.iteration,
                        'params': params.copy(),
                        'loss': loss,
                        'time': iter_time,
                        's_params': s_params_band
                    }
                    self.history.append(iter_data)
                    
                    # 检查是否为最佳结果
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_params = params.copy()
                        self.best_result = s_params_band.copy()
                        print(f"🔥 发现新的最佳结果! 损失: {loss:.4f}")
                    
                    print(f"✅ 迭代完成 | 损失: {loss:.4f} | 耗时: {iter_time:.1f}s")
                    print('='*60)
                    
                    # 生成优化进展图
                    self.plot_current_progress()
                    
                    # 生成当前迭代的S参数曲线图
                    self.plot_iteration_s_params(s_params_band, self.iteration)
                    
                    return loss
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    # 检查是否为超时错误
                    if "超时" in error_msg or "Timeout" in error_msg:
                        self.timeout_count += 1
                        print(f"⏰⏰ 检测到超时 (第 {self.timeout_count} 次)")
                    
                    print(f"❌❌ 迭代 #{self.iteration} 第 {retry_count} 次尝试失败: {error_msg}")
                    
                    if retry_count < max_retries:
                        print(f"🔄🔄 尝试恢复 (剩余重试次数: {max_retries - retry_count})")
                        time.sleep(10)  # 等待10秒
                        
                        # 尝试恢复连接
                        try:
                            self.hfss.close()
                        except:
                            pass
                        
                        # 重新连接
                        if not self.reconnect_hfss():
                            print("⚠️ 重新连接失败")
                        else:
                            print("✅ 重新连接成功")
                            # 更新频率范围和端口映射
                            self.update_frequency_range()
                            self.build_port_name_map()
                    else:
                        print(f"❌❌ 迭代 #{self.iteration} 失败，将损失值设为 {loss}")
            # 记录失败迭代
            iter_time = time.time() - iter_start
            iter_data = {
                'iteration': self.iteration,
                'params': params.copy(),
                'loss': loss,
                'time': iter_time,
                'error': error_msg if 'error_msg' in locals() else "Unknown error"
            }
            self.history.append(iter_data)

            # 定期保存进度
            if self.iteration % 5 == 0:
                self.save_progress()
            
            return loss

        return objective

    def run_simulation_with_timeout(self) -> bool:
        """运行仿真并监控超时"""
        start_time = time.time()
        
        # 创建线程来运行仿真
        class SimulationThread(threading.Thread):
            def __init__(self, hfss_controller):
                super().__init__()
                self.hfss_controller = hfss_controller
                self.success = False
                self.error = None
                
            def run(self):
                try:
                    self.success = self.hfss_controller.analyze()
                except Exception as e:
                    self.error = str(e)
        
        # 启动仿真线程
        sim_thread = SimulationThread(self.hfss)
        sim_thread.start()
        
        # 监控超时
        while sim_thread.is_alive():
            elapsed = time.time() - start_time
            if elapsed > self.iteration_timeout:
                print(f"⏰⏰ 仿真超时! 已运行 {elapsed:.1f}s > {self.iteration_timeout}s")
                try:
                    # 尝试正常关闭仿真
                    print("🛑 尝试中断仿真...")
                    self.hfss._desktop.odesktop.quit_application()
                except:
                    pass
                
                # 强制终止线程
                print("☠️ 强制终止仿真线程...")
                return False
            
            time.sleep(5)  # 每5秒检查一次
        
        # 检查仿真结果
        if sim_thread.error:
            raise RuntimeError(f"仿真错误: {sim_thread.error}")
            
        return sim_thread.success

    def reconnect_hfss(self) -> bool:
        """重新连接HFSS"""
        # 确保关闭现有连接
        try:
            self.hfss.close()
            print("✅ 已关闭HFSS连接")
        except:
            print("⚠️ 关闭HFSS时出错")
        
        # 强制清除锁定文件
        lock_file = self.project_path + ".lock"
        if os.path.exists(lock_file):
            print(f"🔓 尝试清除锁文件: {lock_file}")
            try:
                os.remove(lock_file)
                print("✅ 锁文件已清除")
            except:
                print("⚠️ 无法清除锁文件")
        
        # 重新初始化HFSS控制器
        try:
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
                
            return True
        except Exception as e:
            print(f"❌❌ 重新连接失败: {str(e)}")
            return False

    def calculate_constraint_loss(self, s_params: pd.DataFrame) -> float:
        """计算约束损失函数"""
        total_loss = 0.0
        
        print("\n📉📉 约束损失计算详情:")
        for constraint in self.constraints:
            expr = constraint['expression']
            target = constraint['target']
            operator = constraint['operator']
            weight = constraint['weight']
            freq_range = constraint.get('freq_range')
            freq_point = constraint.get('freq_point')
            aggregate = constraint['aggregate']

            
            # 筛选数据
            if freq_range:
                # 频率范围约束
                freq_min_ghz = freq_range[0] / 1e9
                freq_max_ghz = freq_range[1] / 1e9
                df_sub = s_params[
                    (s_params['Frequency'] >= freq_min_ghz) & 
                    (s_params['Frequency'] <= freq_max_ghz)
                ]
            elif freq_point:
                # 频率点约束
                freq_ghz = freq_point / 1e9
                idx = (s_params['Frequency'] - freq_ghz).abs().idxmin()
                df_sub = s_params.loc[[idx]]
            else:
                # 使用整个频段
                df_sub = s_params
            
            if df_sub.empty:
                # 没有数据，添加大惩罚
                constraint_loss = 100 * weight
                print(f"  ⚠️ 约束 '{expr}' 无有效数据，损失: {constraint_loss:.4f}")
                total_loss += constraint_loss
                continue
            
            # 计算表达式的值
            expr_value = self.evaluate_expression(expr, df_sub, aggregate)

            # 使用自适应损失函数
            constraint_loss = self.adaptive_loss(
                actual=expr_value,
                target=target,
                operator=operator,
                weight=weight,
                iteration=self.iteration
            )
            
            total_loss += constraint_loss
            print(f"  约束 '{expr}' {operator} {target:.4f} | "
                  f"实际值: {expr_value:.4f} | "
                  f"损失: {constraint_loss:.4f} | "
                  f"权重: {weight}")
        
        return total_loss

    def adaptive_loss(self, actual, target, operator, weight, iteration):
        # 检查约束是否已满足
        if operator == '<' and actual < target:
            return 0.0  # 完全满足，无惩罚
        elif operator == '>' and actual > target:
            return 0.0  # 完全满足，无惩罚
        
        # 不满足时才计算惩罚
        gap = abs(actual - target)
        
        # 分段惩罚策略（保持原有逻辑）
        if gap > 10:
            response = 0.1 * gap ** 2
        elif gap > 1:
            response = gap
        else:
            response = 0.5 * gap
            
        return weight * response

    def evaluate_expression(self, expr: str, df: pd.DataFrame, aggregate: str) -> float:
        """在数据框上计算S参数表达式的值"""
        try:
            # 预处理表达式 - 支持 dB(Sxx) 语法
            expr_modified = expr
            for sp in set(re.findall(r'dB\(S\d+\)', expr)):
                sp_name = sp[3:-1]  # 提取 Sxx
                if sp_name in self.global_port_map:
                    ports = self.global_port_map[sp_name]
                    col_name = f"dB(S({ports[0]},{ports[1]}))"
                    expr_modified = expr_modified.replace(sp, f'df["{col_name}"]')
            
            # 替换标准S参数
            for sp in set(re.findall(r'S\d+', expr_modified)):
                if sp in self.global_port_map:
                    ports = self.global_port_map[sp]
                    col_name = f"S({ports[0]},{ports[1]})"
                    expr_modified = expr_modified.replace(sp, f'df["{col_name}"]')
            
            # 添加对复数操作的支持
            expr_modified = expr_modified.replace("abs(", "np.abs(")
            expr_modified = expr_modified.replace("angle(", "np.angle(")
            expr_modified = expr_modified.replace("real(", "np.real(")
            expr_modified = expr_modified.replace("imag(", "np.imag(")
            
            # 添加常用数学函数
            expr_modified = expr_modified.replace("log10(", "np.log10(")
            expr_modified = expr_modified.replace("max(", "np.max(")
            expr_modified = expr_modified.replace("min(", "np.min(")
            expr_modified = expr_modified.replace("mean(", "np.mean(")
            
            # 安全计算表达式
            # 注意：这里使用eval可能存在安全风险，但因为我们控制表达式来源，所以可以接受
            values = eval(expr_modified, {'np': np, 'df': df}, {})
            # 检查计算结果是否为空
            if values is None or (hasattr(values, '__len__') and len(values) == 0):
                raise RuntimeError(f"表达式 '{expr}' 计算结果为空")

            # 应用聚合
            if aggregate == 'min':
                return np.min(values)
            elif aggregate == 'max':
                return np.max(values)
            else:  # 'mean'
                return np.mean(values)
        except Exception as e:
            # 添加详细错误信息
            columns = df.columns.tolist()
            raise RuntimeError(
                f"计算表达式 '{expr}' 失败: {str(e)}\n"
                f"替换后表达式: {expr_modified}\n"
                f"可用列名: {columns}"
            )
    
    def optimize(self, optimizer_type=None):
        """运行优化过程"""
        print(f"\n{'='*60}")
        print("🚀🚀🚀🚀🚀🚀🚀🚀 启动 HFSS 高级约束优化")
        print(f"约束表达式: {[c['expression'] for c in self.constraints]}")
        print(f"优化变量: {[v['name'] for v in self.variables]}")
        print(f"最大迭代次数: {self.max_iter}")
        print('='*60)
        
        self.start_time = time.time()
        
        try:
            # 初始化HFSS控制器
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
            
            # 更新频率范围到目标频段
            if not self.update_frequency_range():
                raise RuntimeError("无法设置目标频率范围")
            
            # 获取优化前的端口信息
            ports = self.hfss.get_ports()
            print(f"🔌🔌 检测到的端口: {ports}")
            
            # 创建目标函数
            objective_func = self.create_objective_function()
            # 替换优化算法部分
            if optimizer_type == "cmaes":
                from cmaes import CMA
                print("🔄🔄 使用CMA-ES优化器...")
                optimizer = CMA(
                    mean=np.array([(v['bounds'][0] + v['bounds'][1])/2 for v in self.variables]),
                    sigma=0.3,  # 初始步长
                    population_size=1, # min(10, self.max_iter//5),  # 自适应种群大小
                    bounds=np.array([(v['bounds'][0], v['bounds'][1]) for v in self.variables])
                )

                no_improve_count = 0
                best_loss = float('inf')

                for generation in range(self.max_iter):
                    solutions = []
                    for _ in range(optimizer.population_size):
                        x = optimizer.ask()
                        loss = objective_func(x)
                        solutions.append((x, loss))
                        # 记录历史
                        self.iteration += 1
                    
                    # 更新优化器
                    optimizer.tell(solutions)

                    # 找到当前代的最佳解
                    current_best_idx = np.argmin([s[1] for s in solutions])
                    current_best_loss = solutions[current_best_idx][1]

                    # 更新全局最佳
                    if current_best_loss < best_loss:
                        # 有显著改进（>1%）
                        if (best_loss - current_best_loss) / best_loss > 0.01:
                            no_improve_count = 0
                        best_loss = current_best_loss
                        self.best_loss = best_loss
                        self.best_params = dict(zip(
                            [v['name'] for v in self.variables], 
                            solutions[current_best_idx][0]
                        ))
                    else:
                        no_improve_count += 1
                    
                    # 早停机制
                    if no_improve_count >= 3:
                        print(f"🚩 早停触发：连续{no_improve_count}代无显著改进")
                        break
                        
                result = self.best_params
                
            elif optimizer_type == "de":
                from scipy.optimize import differential_evolution
                print("🔄🔄 使用差分进化优化器...")
                bounds = [(v['bounds'][0], v['bounds'][1]) for v in self.variables]
                result = differential_evolution(
                    func=objective_func,
                    bounds=bounds,
                    maxiter=self.max_iter//10,  # 代数
                    popsize=10,  # 种群大小
                    mutation=(0.5, 1.0),  # 自适应变异
                    recombination=0.9,
                    strategy='best1bin',
                    tol=0.01
                )
            
            elif optimizer_type == "pso":
                result = self.pso_optimize(objective_func)

            else:
                # 运行贝叶斯优化
                print("🔄🔄 开始贝叶斯优化...")
                result = gp_minimize(
                    func=objective_func,
                    dimensions=[Real(v['bounds'][0], v['bounds'][1]) for v in self.variables],
                    n_calls=self.max_iter,
                    random_state=42,
                    acq_func='EI',  # 期望改进
                    base_estimator='RF', # 'RF':随机森林, 'GP':高斯过程, 'ET':极端随机树, 'GBRT':梯度提升回归树
                    n_initial_points=min(50, max(20,5*len(self.variables))),
                    n_jobs=-1,
                    acq_optimizer='sampling',  # 'sampling':随机采样, 'lbfgs':L-BFGS-B, 'gbrt':梯度提升回归树, 'auto':自动选择
                    verbose=True
                )
            
            # 保存最终优化结果
            self.save_results(result)
            
            # 可视化结果
            self.visualize_results(result)
            
            return result
            
        except Exception as e:
            print(f"❌❌❌❌ 优化失败: {str(e)}")
            traceback.print_exc()
            return None
        finally:
            # 最终结果汇总
            print("\n" + "="*60)
            print(f"🏁🏁 优化完成! 最佳损失: {self.best_loss:.4f}")
            if self.best_params:
                print("最佳参数:")
                for name, value in self.best_params.items():
                    print(f"  {name}: {value:.4f}")
            
            if self.hfss:
                self.hfss.close()
            total_time = (time.time() - self.start_time) / 60
            print(f"\n⏱⏱⏱ 总优化时间: {total_time:.1f} 分钟")

    def pso_optimize(self, objective_func):
        """改进型粒子群优化算法(PSO)"""
        print("🔄🔄🔄🔄 使用改进型粒子群优化(PSO)...")
        
        # PSO参数配置
        n_particles = min(20, max(10, 5 * len(self.variables)))  # 粒子数量
        max_iter = self.max_iter  # 最大迭代次数
        
        # 变量边界
        bounds = np.array([(v['bounds'][0], v['bounds'][1]) for v in self.variables])
        dim = len(bounds)
        
        # 改进型PSO参数
        w_max = 0.9  # 最大惯性权重
        w_min = 0.4  # 最小惯性权重
        c1_max = 2.5  # 最大个体学习因子
        c1_min = 1.0  # 最小个体学习因子
        c2_max = 2.5  # 最大社会学习因子
        c2_min = 1.0  # 最小社会学习因子
        mutation_prob = 0.2  # 突变概率
        convergence_threshold = 1e-5  # 收敛阈值
        no_improve_limit = 10  # 无改进迭代次数限制
        
        # 初始化粒子群
        particles = np.random.uniform(
            low=bounds[:, 0], 
            high=bounds[:, 1], 
            size=(n_particles, dim)
        )
        velocities = np.zeros((n_particles, dim))
        
        # 初始化个体最优位置和适应度
        personal_best_positions = np.copy(particles)
        personal_best_fitness = np.array([objective_func(p) for p in particles])
        
        # 初始化全局最优
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = np.copy(personal_best_positions[global_best_idx])
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        # 记录最佳适应度历史
        best_fitness_history = [global_best_fitness]
        no_improve_count = 0
        
        # 优化循环
        for iter in range(max_iter):
            # 自适应参数调整
            w = w_max - (w_max - w_min) * iter / max_iter
            c1 = c1_max - (c1_max - c1_min) * iter / max_iter
            c2 = c2_min + (c2_max - c2_min) * iter / max_iter
            
            # 更新粒子位置和速度
            for i in range(n_particles):
                # 更新速度
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 c2 * r2 * (global_best_position - particles[i]))
                
                # 位置更新
                particles[i] += velocities[i]
                
                # 边界处理
                particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
                
                # 评估适应度
                fitness = objective_func(particles[i])
                
                # 更新个体最优
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = np.copy(particles[i])
                    
                    # 更新全局最优
                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = np.copy(particles[i])
                        no_improve_count = 0  # 重置无改进计数器
                        print(f"🔥 PSO迭代 {iter+1}/{max_iter}: 发现新全局最优 {global_best_fitness:.4f}")
            
            # 记录最佳适应度
            best_fitness_history.append(global_best_fitness)
            
            # 检查收敛性
            if iter > 0:
                improvement = best_fitness_history[-2] - best_fitness_history[-1]
                if improvement < convergence_threshold:
                    no_improve_count += 1
                    print(f"⏳ PSO迭代 {iter+1}/{max_iter}: 改进量 {improvement:.6f} < 阈值")
                else:
                    no_improve_count = 0
            
            # 早停机制
            if no_improve_count >= no_improve_limit:
                print(f"🛑 PSO早停: 连续 {no_improve_count} 次迭代无显著改进")
                break
                
            # 突变机制 - 避免局部最优
            if iter % 5 == 0 and global_best_fitness > 0.1:  # 只有当适应度不够好时才突变
                print("🧬 执行突变操作...")
                for i in range(n_particles):
                    if np.random.rand() < mutation_prob:
                        # 对粒子位置进行随机扰动
                        mutation_strength = 0.2 * (bounds[:, 1] - bounds[:, 0])
                        particles[i] += np.random.normal(0, mutation_strength, dim)
                        particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
                        
                        # 重新评估并更新
                        fitness = objective_func(particles[i])
                        if fitness < personal_best_fitness[i]:
                            personal_best_fitness[i] = fitness
                            personal_best_positions[i] = np.copy(particles[i])
                            
                            if fitness < global_best_fitness:
                                global_best_fitness = fitness
                                global_best_position = np.copy(particles[i])
                                print(f"🧪 突变后发现新全局最优 {global_best_fitness:.4f}")
            
            # 更新优化器状态
            self.best_loss = global_best_fitness
            self.best_params = dict(zip([v['name'] for v in self.variables], global_best_position))
        
        # 创建优化结果对象
        class PSO_Result:
            def __init__(self, x, fun):
                self.x = x
                self.fun = fun
                self.success = True
                self.message = "PSO optimization completed"
        
        return PSO_Result(global_best_position, global_best_fitness)

    def save_results(self, result):
        """保存优化结果"""
        # 创建时间戳目录
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join(self.output_dir, f"optim_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 转换numpy数组为可序列化格式
        optim_result = {
            'x': result.x.tolist() if isinstance(result.x, np.ndarray) else result.x,
            'fun': result.fun,
            'x_iters': [x.tolist() if isinstance(x, np.ndarray) else x for x in result.x_iters],
            'func_vals': result.func_vals.tolist() if isinstance(result.func_vals, np.ndarray) else result.func_vals
        }

        # 保存优化结果
        result_data = {
            'project': self.project_path,
            'design': self.design_name,
            'setup': self.setup_name,
            'sweep': self.sweep_name,
            'variables': self.variables,
            'constraints': self.constraints,
            'freq_range': self.freq_range,
            'best_params': self.best_params,
            'best_loss': self.best_loss,
            'optimizer_result': optim_result
        }
        
        with open(os.path.join(self.save_dir, 'optim_result.json'), 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # 保存优化历史
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(self.save_dir, 'optim_history.csv'), index=False)
        
        # 保存最佳S参数结果
        if self.best_result is not None:
            self.best_result.to_csv(os.path.join(self.save_dir, 'best_s_params.csv'), index=False)
        
        print(f"💾💾 优化结果已保存至: {self.save_dir}")
    
    def visualize_results(self, result):
        """可视化优化结果"""

        plt.figure(figsize=(15, 10))
        
        # 收敛曲线
        plt.subplot(2, 2, 1)
        plot_convergence(result)
        plt.title("优化收敛曲线")
        
        # 目标函数评估
        plt.subplot(2, 2, 2)
        plt.plot(result.func_vals)
        plt.xlabel("迭代次数")
        plt.ylabel("损失值")
        plt.title("损失值变化")
        plt.grid(True)
        
        # 最佳S参数曲线
        if self.best_result is not None and not self.best_result.empty:
            plt.subplot(2, 2, 3)
            
            # 收集所有需要绘图的S参数
            sparams_to_plot = set()
            for constraint in self.constraints:
                sparams_to_plot.update(re.findall(r'[a-zA-Z0-9_]+\(S\d+\)|S\d+', constraint['expression']))
            
            # 绘制曲线
            for sp in sparams_to_plot:
                # 处理带函数的S参数 (如 dB(S11))
                is_dB = 'dB(' in sp
                sp_name = sp[3:-1] if is_dB else sp  # 提取 Sxx
                
                if sp_name in self.global_port_map:
                    ports = self.global_port_map[sp_name]
                    col_name = f"dB(S({ports[0]},{ports[1]}))" if is_dB else f"S({ports[0]},{ports[1]})"
                    
                    if col_name in self.best_result.columns:
                        # 对于复数S参数，绘制幅值
                        if not is_dB and 'complex' in str(self.best_result[col_name].dtype):
                            # 计算幅值 (dB)
                            magnitude = 20 * np.log10(np.abs(self.best_result[col_name]))
                            plt.plot(
                                self.best_result['Frequency'], 
                                magnitude,
                                label=f"{sp_name} (幅值)"
                            )
                        else:
                            plt.plot(
                                self.best_result['Frequency'], 
                                self.best_result[col_name],
                                label=f"{sp} (dB)" if is_dB else f"{sp_name} (dB)"
                            )
            
            plt.xlabel("频率 (GHz)")
            plt.ylabel("S参数 (dB)")
            plt.title("最佳S参数")
            plt.grid(True)
            plt.legend()
            plt.xlim(self.freq_range[0]/1e9, self.freq_range[1]/1e9)

            plt.savefig(os.path.join(self.output_dir, "optimization.png"))
        
        # 参数重要性
        plt.subplot(2, 2, 4)
        try:
            plot_objective(result)
            plt.title("参数重要性")
        except:
            plt.text(0.5, 0.5, "参数重要性图不可用", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "optimization_summary.png"))
        plt.show()
        
        print("📊📊 优化结果可视化完成")

    def plot_current_progress(self):
        """生成并保存当前优化进展图"""
        if not self.history:
            return
            
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制损失曲线
        iterations = [h['iteration'] for h in self.history]
        losses = [h['loss'] for h in self.history]
        
        plt.plot(iterations, losses, 'bo-', label='总损失')
        
        # 标记最佳点
        best_idx = np.argmin(losses)
        best_iter = iterations[best_idx]
        best_loss = losses[best_idx]
        plt.plot(best_iter, best_loss, 'ro', markersize=8, label='最佳点')

        # 添加最佳点标注
        plt.annotate(f'迭代 #{best_iter}\n损失: {best_loss:.4f}',
                    xy=(best_iter, best_loss),
                    xytext=(best_iter + 1, best_loss + 0.1 * max(losses)),
                    arrowprops=dict(facecolor='red', shrink=0.05))

        # 添加最佳参数值
        if self.best_params:
            param_text = "最佳参数:\n"
            for name, value in self.best_params.items():
                param_text += f"{name}: {value:.4f}\n"
            
            plt.figtext(0.75, 0.25, param_text, 
                    bbox=dict(facecolor='lightgreen', alpha=0.5),
                    fontsize=9)
        
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title(f'优化进展 (当前迭代: {self.iteration}/{self.max_iter})')
        plt.grid(True)
        plt.legend()
        
        # 保存图片
        os.makedirs(os.path.join(self.output_dir, "progress_plots"), exist_ok=True)
        plot_path = os.path.join(self.output_dir, "progress_plots", f"progress_iter_{self.iteration}.png")
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path

    def build_port_name_map(self):
        """构建端口名称映射：将数字映射到实际端口名称"""
        # 获取实际检测到的端口
        ports = self.hfss.get_ports()
        print(f"🔌🔌🔌🔌 实际检测到的端口: {ports}")
        
        # 创建映射：数字 → 完整端口名称
        self.port_name_map = {}
        for p in ports:
            # 提取端口数字部分（保持原始格式）
            port_num = p
            self.port_name_map[port_num] = p
        
        # 更新全局端口映射中的端口名称
        for sp_name, ports in self.global_port_map.items():
            tx_num, rx_num = ports
            tx_port = self.port_name_map.get(tx_num, tx_num)
            rx_port = self.port_name_map.get(rx_num, rx_num)
            self.global_port_map[sp_name] = (tx_port, rx_port)
        
        # 验证映射完整性
        print(f"🔀🔀🔀🔀 更新后的全局端口映射: {self.global_port_map}")

    def plot_iteration_s_params(self, s_params: pd.DataFrame, iteration: int):
        """绘制当前迭代的S参数曲线"""
        try:
            # 确保在主线程操作
            if threading.current_thread() != threading.main_thread():
                print("⚠️ 绘图操作跳过：非主线程环境")
                return ""
                
            # 添加安全锁
            plot_lock = threading.Lock()
            with plot_lock:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(14, 8))
                
                # 设置颜色映射
                color_map = {
                    'S11': 'red',
                    'S21': 'blue',
                    'S31': 'green',
                    'S41': 'purple',
                    'S51': 'orange',
                    'S61': 'brown'
                }
    
                # 收集所有需要绘图的S参数
                sparams_to_plot = set()
                for constraint in self.constraints:
                    sparams_to_plot.update(re.findall(r'[a-zA-Z0-9_]+\(S\d+\)|S\d+', constraint['expression']))
                
                # 绘制曲线
                for sp in sparams_to_plot:
                    # 处理带函数的S参数 (如 dB(S11))
                    is_dB = 'dB(' in sp
                    sp_name = sp[3:-1] if is_dB else sp  # 提取 Sxx
                    
                    if sp_name in self.global_port_map:
                        # 获取实际端口名称
                        tx_num, rx_num = self.global_port_map[sp_name]
                        tx_port = self.port_name_map.get(tx_num, tx_num)
                        rx_port = self.port_name_map.get(rx_num, rx_num)
                        
                        # 确定列名
                        col_name = f"dB(S({tx_port},{rx_port}))" if is_dB else f"S({tx_port},{rx_port})"
                        
                        if col_name in s_params.columns:
                            # 对于复数S参数，绘制幅值 (dB)
                            if not is_dB and 'complex' in str(s_params[col_name].dtype):
                                # 计算幅值 (dB)
                                magnitude = 20 * np.log10(np.abs(s_params[col_name]))
                                plt.plot(
                                    s_params['Frequency'], 
                                    magnitude,
                                    label=f"{sp_name} (幅值)",
                                    color=color_map.get(sp_name, 'gray'),
                                    linewidth=2
                                )
                            else:
                                plt.plot(
                                    s_params['Frequency'], 
                                    s_params[col_name],
                                    label=f"{sp} ({tx_port}→{rx_port})" if is_dB else f"{sp_name} ({tx_port}→{rx_port})",
                                    color=color_map.get(sp_name, 'gray'),
                                    linewidth=2
                                )
                            
                # 设置图表属性
                plt.title(f"迭代 #{iteration} S参数曲线\n(当前损失: {self.history[-1]['loss']:.4f} | 最佳损失: {self.best_loss:.4f})", 
                        fontsize=14)
                plt.xlabel("频率 (GHz)", fontsize=12)
                plt.ylabel("S参数 (dB)", fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='best', fontsize=10)
                
                # 设置频率范围
                plt.xlim(self.freq_range[0]/1e9, self.freq_range[1]/1e9)
                
                # 添加约束信息
                constraint_info = "\n".join([f"{c['expression']} {c['operator']} {c['target']:.4f}" 
                                            for c in self.constraints])
                plt.figtext(0.75, 0.15, f"约束条件:\n{constraint_info}", 
                        fontsize=9, bbox=dict(facecolor='lightyellow', alpha=0.5))
                
                # 添加参数值信息
                param_info = "\n".join([f"{k}: {v:.4f}" 
                                    for k, v in self.history[-1]['params'].items()])
                plt.figtext(0.75, 0.30, f"当前参数值:\n{param_info}", 
                        fontsize=9, bbox=dict(facecolor='lightblue', alpha=0.5))
                
                # 保存图片
                plot_dir = os.path.join(self.output_dir, "s_params_plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, f"s_params_iter_{iteration}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"📊📊 已生成S参数曲线图: {plot_path}")
                return plot_path
            
        except Exception as e:
            print(f"⚠️ 绘制S参数曲线失败: {str(e)}")
            return ""

def main():
    """主函数 - 优化示例"""
    # 项目配置
    PROJECT_PATH = r"C:\Users\Administrator\Desktop\huaSheng\6G\6G.aedt"
    DESIGN_NAME = "HFSSDesign5"
    SETUP_NAME = "Setup1"
    SWEEP_NAME = "Sweep"
    
    # 全局端口映射
    GLOBAL_PORT_MAP = {
        'S11': ('1', '1'),
        #'S21': ('2', '1'),
        #'S31': ('3', '1'),
        #'S41': ('4', '1')
    }
    
    # 约束配置 - 高级示例
    CONSTRAINTS = [
        #{
        #    'expression': 'dB(S21) - dB(S31)',
        #    'target': 0.5,
        #    'operator': '<',  # S21与S31的dB差值小于0.5 dB
        #    'weight': 1.0,
        #    'freq_range': (5.5e9, 7.5e9),
        #    'aggregate': 'max'
        #},
        {
            'expression': 'mean(dB(S11))',  # 均方误差更平滑
            'target': -13,  # 比目标值低3dB的裕量
            'operator': '<', 
            'weight': 0.4,
            'freq_range': (5.9e9, 7.2e9),
            'aggregate': 'mean'
        },
        {
            'expression': 'dB(S11)',
            'target': -11,
            'operator': '<',  # 所有端口的最大反射系数小于-10 dB
            'weight': 0.6,
            'freq_range': (5.9e9, 6.5e9),
            'aggregate': 'max'
        },
        {
            'expression': 'dB(S11)',
            'target': -11,
            'operator': '<',  # 所有端口的最大反射系数小于-10 dB
            'weight': 0.6,
            'freq_range': (6.5e9, 7.2e9),
            'aggregate': 'max'
        },
        #{
        #    'expression': 'min(dB(S21), dB(S31))',
        #    'target': -2.0,
        #    'operator': '>',  # S21和S31的最小值大于-2 dB
        #    'weight': 1.0,
        #    'freq_point': 6.0e9
        #},
        #{
        #    'expression': 'abs(angle(S21) - angle(S31))',
        #    'target': 10,  # 相位差小于10度
        #    'operator': '<',
        #    'weight': 0.8,
        #    'freq_range': (5.5e9, 7.5e9),
        #    'aggregate': 'max'
        #}
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
    
    # 创建并运行优化器
    optimizer = HfssAdvancedConstraintOptimizer(
        project_path=PROJECT_PATH,
        design_name=DESIGN_NAME,
        setup_name=SETUP_NAME,
        sweep_name=SWEEP_NAME,
        variables=VARIABLES,
        freq_range=(4.5e9, 8e9),
        constraints=CONSTRAINTS,
        global_port_map=GLOBAL_PORT_MAP,
        max_iter=50
    )
    
    # 开始优化
    result = optimizer.optimize(optimizer_type='pso')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌❌❌❌ 程序异常: {str(e)}")
        traceback.print_exc()