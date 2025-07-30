"""HFSS API 接口库 - 使用 PyAEDT 实现 Ansys HFSS 的自动化控制
主要功能：变量修改、运行仿真、获取 S 参数结果
设计原则：简洁性、稳定性、可维护性
"""
import os
import time
import psutil
import traceback
import numpy as np
import pandas as pd
from ansys.aedt.core import Hfss
import time

class HFSSController:
    """HFSS 自动化控制接口
    
    通过上下文管理器管理 HFSS 会话生命周期，确保资源正确释放：
    with HFSSController(...) as hfss:
        # 使用 hfss 对象
    """
    
    def __init__(self, project_path, design_name="HFSSDesign1", 
                 setup_name="Setup1", sweep_name="Sweep", port=54100,
                 default_length_unit='mm', default_angle_unit="deg"):
        """
        初始化 HFSS 控制器
        
        :param project_path: HFSS 项目路径 (.aedt)
        :param design_name: 设计名称 (默认: "HFSSDesign1")
        :param setup_name: 仿真设置名称 (默认: "Setup1")
        :param sweep_name: 扫频名称 (默认: "Sweep")
        :param port: gRPC 端口 (默认: 54100)
        :param default_length_unit: 默认长度单位 (默认: "mm")
        :param default_angle_unit: 默认角度单位 (默认: "deg")
        """
        self.project_path = project_path
        self.lock_file = project_path + ".lock"
        self.design_name = design_name
        self.setup_name = setup_name
        self.sweep_name = sweep_name
        self.port = port
        self.default_length_unit = default_length_unit
        self.default_angle_unit = default_angle_unit
        self.hfss = None
        self._desktop = None
        self.model_units = None  # 存储模型单位
    
    def _force_unlock_file(self, file_path):
        """强制解除文件锁定
       
        当检测到锁文件时，尝试终止占用进程并删除锁文件
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ 已清除锁文件: {file_path}")
                return True
        except PermissionError:
            print("⚠️ 尝试终止占用进程...")
            for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                try:
                    # 查找占用锁文件的 ANSYS 进程
                    if "ansysedt.exe" in proc.info['name'].lower():
                        for file in proc.info.get('open_files', []):
                            if file_path.lower() in file.path.lower():
                                print(f"终止进程: PID={proc.pid}, 名称={proc.info['name']}")
                                proc.kill()
                                time.sleep(2)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                    continue
            print("❌ 删除失败：请重启电脑后手动删除锁文件")
        except Exception as e:
            print(f"❌ 解锁文件错误: {str(e)}")
        return False
    
    def connect(self):
        """连接到 HFSS 并打开项目

        返回: True 连接成功, False 连接失败
        """
        try:
            # 清除可能存在的锁文件
            if os.path.exists(self.lock_file):
                print("⚠️ 检测到锁文件，尝试清除...")
                self._force_unlock_file(self.lock_file)
            
            # 创建 HFSS 会话
            print("🚀 启动 HFSS 会话...")
            self.hfss = Hfss(
                project=self.project_path,
                design=self.design_name,
                version="2023.1",
                new_desktop=True,
                close_on_exit=False,
                port=self.port
            )
            self._desktop = self.hfss._desktop
            
            # 获取并存储模型单位
            self.model_units = self.hfss.modeler.model_units
            print(f"🔗 已连接项目: {self.hfss.project_name} (单位: {self.model_units})")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def check_design_config(self):
        """检查设计配置是否有效
        
        验证 setup 和 sweep 是否存在
        返回: True 配置有效, False 配置无效
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            print("\n📋 设计配置检查:")
            
            # 1. 检查 Setup 是否存在
            setup_names = [setup.name for setup in self.hfss.setups]
            print(f"  可用 Setup 列表: {setup_names}")
            if self.setup_name not in setup_names:
                raise ValueError(f"❌ 未找到 Setup: {self.setup_name}（可用：{setup_names}）")
            
            # 2. 检查 Sweep 是否存在
            setup = self.hfss.get_setup(self.setup_name)
            if not setup:
                raise ValueError(f"❌ 无法获取 Setup 对象: {self.setup_name}")
            
            sweep_names = [sweep.name for sweep in setup.sweeps]
            print(f"  {self.setup_name} 下的 Sweep 列表: {sweep_names}")
            
            # 更新扫频名称（如果找不到则使用第一个）
            if sweep_names:
                if self.sweep_name not in sweep_names:
                    print(f"⚠️ 未找到指定 Sweep: {self.sweep_name}，使用第一个可用 Sweep: {sweep_names[0]}")
                    self.sweep_name = sweep_names[0]
            else:
                print("⚠️ 未找到任何 Sweep，将直接使用 Setup")
                self.sweep_name = None
            
            return True
        except Exception as e:
            print(f"❌ 设计配置检查失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_ports(self):
        """获取所有端口名称
        
        返回: 端口名称列表
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            ports = []
            try:
                # 方法1: 使用标准属性
                ports = self.hfss.excitations
            except AttributeError:
                try:
                    # 方法2: 使用备用方法
                    ports = self.hfss.get_excitations()
                except Exception:
                    print("⚠️ 使用备用方法获取端口失败")
            
            # 如果以上方法都失败，尝试常见端口名称
            if not ports:
                port_candidates = ["1", "Port1", "Port_1", "P1"]
                for candidate in port_candidates:
                    try:
                        # 检查端口是否存在
                        if candidate in self.hfss.get_excitations():
                            ports = [candidate]
                            break
                    except Exception:
                        continue
            
            # 确保至少返回一个端口
            if not ports:
                ports = ["1"]  # 默认值
                print("⚠️ 使用默认端口 '1'")
            
            print(f"✅ 获取端口列表: {ports}")
            return ports
        except Exception as e:
            print(f"❌ 获取端口失败: {str(e)}")
            return ["1"]  # 默认值
    
    def set_variable(self, variable_name, value, unit=None):
        """
        设置变量值（带单位支持）
        
        :param variable_name: 变量名称
        :param value: 新值
        :param unit: 单位 (如 "mm", "deg", "GHz"等)
        返回: True 设置成功, False 设置失败
        """
        # 添加类型验证
        if isinstance(value, (list, np.ndarray)):
            raise TypeError(f"❌ 变量值必须是标量，当前是{type(value)}: {value}")
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            # 智能推断单位类型
            var_lower = variable_name.lower()
            if unit is None:
                if any(kw in var_lower for kw in ["length", "width", "height", "radius", "thick"]):
                    unit = self.model_units if self.model_units else self.default_length_unit
                elif any(kw in var_lower for kw in ["angle", "theta", "phi"]):
                    unit = self.default_angle_unit
                else:
                    unit = ""  # 无量纲量
            
            # 格式化带单位的数值
            value_str = f"{value}{unit}" if unit else str(value)
            '''
            # 使用变量管理器安全设置变量
            var_manager = self.hfss.variable_manager
            if variable_name in var_manager.variables:
                var_manager.set_variable_value(variable_name, value_str)
            else:
                var_manager.set_variable(variable_name, value_str)
            '''
            # 使用更兼容的变量设置方法
            self.hfss.variable_manager[variable_name] = value_str
            print(f"✅ 设置变量 {variable_name} = {value_str}")
            return True
        except Exception as e:
            print(f"❌ 设置变量失败: {str(e)}")
            return False
    
    def analyze(self):
        """运行仿真
        
        返回: True 仿真成功, False 仿真失败
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            print(f"\n🚀 启动仿真: {self.setup_name}...")
            start_time = time.time()
            
            # 执行仿真
            self.hfss.analyze_setup(self.setup_name)
            
            elapsed = time.time() - start_time
            print(f"✅ 仿真完成! 耗时: {elapsed:.2f}秒")
            return True
        except Exception as e:
            print(f"❌ 仿真失败: {str(e)}")
            return False
    
    def get_s_params(self, port_combinations=None, batch_size=1, data_format="both"):
        """
        获取 S 参数结果 (更稳定的实现)
        
        使用 PyAEDT 的报告生成功能获取 S 参数
        
        :param port_combinations: 端口组合列表，如 [('1','1'), ('1','2')]
        :param batch_size: 此参数保留但不再使用（为了接口兼容）
        :param data_format: 数据格式 ("dB" - 仅dB格式, "complex" - 仅复数格式, "both" - 两者都获取)
        返回: 包含所有 S 参数的 DataFrame
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            
            # 确定扫频路径
            sweep_path = f"{self.setup_name} : {self.sweep_name}" if self.sweep_name else self.setup_name
            
            print(f"🔍🔍 获取 S 参数矩阵 (扫频路径: {sweep_path})")
            
            # 获取所有端口
            ports = self.get_ports()
            port_names = sorted(ports)  # 确保端口顺序一致
            
            # 如果没有指定端口组合，生成所有可能的组合
            if port_combinations is None:
                port_combinations = [(p1, p2) for p1 in port_names for p2 in port_names]
            
            # 创建结果 DataFrame
            result_df = pd.DataFrame()
            
            # 创建报告对象 (使用新的参数名 setup)
            report = self.hfss.post.reports_by_category.standard(setup=sweep_path)
            if not report:
                print("❌❌ 无法创建报告对象")
                return None
                
            # 设置报告频率扫描
            report.domain = "Sweep"
            
            # 设置报告表达式
            expressions = []
            for tx, rx in port_combinations:
                # 创建标准化的表达式名称
                complex_expr = f"S({tx},{rx})".replace(" ", "")
                expressions.append(complex_expr)
                
                if data_format in ["dB", "both"]:
                    db_expr = f"dB(S({tx},{rx}))"
                    expressions.append(db_expr)
            
            # 添加表达式到报告
            report.expressions = expressions
            
            # 创建频率点数组 - 使用新的属性
            solution = self.hfss.setups[0].sweeps[0] if self.sweep_name else self.hfss.setups[0]
            frequencies = solution.frequencies  # 修改这里
            
            # 获取报告数据
            report_data = report.get_solution_data()
            if report_data is None:
                print("❌❌ 无法获取报告数据")
                return None
                
            # 添加频率数据到DataFrame
            result_df["Frequency"] = frequencies
            
            # 处理每个表达式
            for expr in expressions:
                try:
                    # 对于复数表达式
                    if expr.startswith('S(') and 'dB' not in expr:
                        # 正确获取实部和虚部
                        real_part = report_data.data_real(expr)
                        imag_part = report_data.data_imag(expr)
                        
                        if real_part is not None and imag_part is not None:
                            # 组合实部和虚部形成复数
                            expr_complex = [complex(real, imag) for real, imag in zip(real_part, imag_part)]
                            result_df[expr] = expr_complex
                            print(f"✅ 获取复数格式成功: {expr}")
                        else:
                            # 如果无法获取虚部，尝试直接获取复数数据
                            try:
                                expr_complex = report_data.data_complex(expr)
                                if expr_complex is not None:
                                    result_df[expr] = expr_complex
                                    print(f"✅ 直接获取复数格式成功: {expr}")
                                else:
                                    print(f"⚠️ 无法获取复数数据: {expr}")
                            except:
                                print(f"⚠️ 无法获取复数数据: {expr}")
                    
                    # 对于dB表达式
                    elif expr.startswith('dB'):
                        # 正确获取dB值
                        db_data = report_data.data_real(expr)
                        if db_data is not None:
                            # 确保数据是浮点数格式
                            result_df[expr] = [float(val) for val in db_data]
                            print(f"✅ 获取dB格式成功: {expr}")
                        else:
                            print(f"⚠️ 无法获取dB数据: {expr}")
                            
                except Exception as e:
                    print(f"❌❌ 处理表达式 {expr} 失败: {str(e)}")
                    traceback.print_exc()
            
            # 数据预览
            if not result_df.empty:
                print("\n📊📊 S 参数数据预览:")
                print(result_df.head(3))
                print(f"  数据点数: {len(result_df)}")
                print(f"  参数数量: {len(result_df.columns) - 1}")
                
                # 添加复数数据验证
                complex_cols = [col for col in result_df.columns 
                            if col.startswith('S(') and 'dB' not in col]
                if complex_cols:
                    print("\n复数S参数验证:")
                    for col in complex_cols:
                        sample = result_df[col].iloc[0]
                        # 验证类型和值
                        if isinstance(sample, complex):
                            print(f"  {col}: complex 示例: {sample}")
                        elif isinstance(sample, float):
                            print(f"  {col}: float 示例: {sample}")
                        else:
                            print(f"  {col}: 未知类型 {type(sample)}")
                else:
                    print("⚠️ 未检测到复数格式S参数数据")
            else:
                print("❌❌ 未获取到有效数据")
                
            return result_df

        except Exception as e:
            print(f"❌❌ 获取 S 参数失败: {str(e)}")
            traceback.print_exc()
            return None

    def save_s_params(self, s_params, output_csv=None):
        """保存原始S参数数据到CSV文件"""
        if output_csv is None:
            import tempfile
            output_csv = os.path.join(
                tempfile.gettempdir(),
                f"{os.path.basename(self.project_path).replace('.aedt', '')}_s_params.csv"
            )
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            
            # 保存为CSV
            s_params.to_csv(output_csv, index=False)
            print(f"💾💾 原始S参数已保存至: {output_csv}")
            return output_csv
        except Exception as e:
            print(f"❌❌ 保存S参数失败: {str(e)}")
            return None
            
    def save_project(self, new_path=None):
        """保存项目

        :param new_path: 可选的新路径
        返回: True 保存成功, False 保存失败
        """
        try:
            if not self.hfss:
                raise RuntimeError("未连接到 HFSS，请先调用 connect()")
            if new_path:
                self.hfss.save_project(new_path)
                print(f"💾 项目已另存为: {new_path}")
            else:
                self.hfss.save_project()
                print("💾 项目已保存")
            return True
        except Exception as e:
            print(f"❌ 保存失败: {str(e)}")
            return False

    def close(self):
        """关闭 HFSS 连接

        返回: True 关闭成功, False 关闭失败
        """
        try:
            # 先释放matplotlib资源
            import matplotlib.pyplot as plt
            plt.close('all')
            # 再关闭HFSS连接
            if self.hfss:
                print("🛑 正在关闭 ANSYS...")
                self.hfss.close_desktop()
                print("✅ ANSYS 已关闭")
                self.hfss = None
                self._desktop = None
                # 添加延迟确保资源释放
                time.sleep(5)
            return True
    
        except Exception as e:
            print(f"❌ 关闭失败: {str(e)}")
            return False

    def export_results(self, df, output_csv=None, max_retries=3):
        """导出结果到CSV文件"""
        try:
            if output_csv is None:
                import tempfile
                output_csv = os.path.join(
                    tempfile.gettempdir(),
                    os.path.basename(self.project_path).replace(".aedt", "_results.csv")
                )
            
            # 确保输出路径是文件而非目录
            if os.path.isdir(output_csv):
                output_csv = os.path.join(output_csv, "hfss_results.csv")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            
            for i in range(max_retries):
                try:
                    df.to_csv(output_csv, index=False)
                    print(f"💾 结果已导出至: {output_csv}")
                    return output_csv
                except PermissionError as pe:
                    if i < max_retries - 1:
                        print(f"⚠️ 文件占用中，等待重试 ({i+1}/{max_retries})...")
                        time.sleep(30)  #等待30秒
                    else:
                        print(f"❌ 多次尝试失败: {str(pe)}")
                        return None
        except Exception as e:
            print(f"❌ 导出结果失败: {str(e)}")
            return None

    def __enter__(self):
        """上下文管理器入口 - 自动连接"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """上下文管理器出口 - 自动关闭"""
        self.close()