import os
import sys
import datetime
import time
import subprocess
import signal
import threading

class TaskMonitor:
    def __init__(self):
        self.active_tasks = {}
        self.lock = threading.Lock()
        self.completed_tasks = set()
    
    def add_task(self, task_id, process_info):
        with self.lock:
            self.active_tasks[task_id] = process_info
    
    def mark_completed(self, task_id):
        with self.lock:
            if task_id in self.active_tasks:
                self.completed_tasks.add(task_id)
                return True
            return False
    
    def get_active_tasks(self):
        with self.lock:
            return list(self.active_tasks.keys())
    
    def get_task_info(self, task_id):
        with self.lock:
            return self.active_tasks.get(task_id, None)
    
    def all_completed(self):
        with self.lock:
            return len(self.completed_tasks) == len(self.active_tasks)

def run_parallel_evaluations():
    # ===================== 参数配置 =====================
    num_gpus = 4  # 可用的GPU数量
    
    # NOTE: 模型检查点配置 (ckpt, config, log_path)
    ckpt_paths = [
        (
            "path/to/VLA-Checkpoint-{epoch}-{steps}.ckpt/or/VLA-Checkpoint.pt",
            "path/to/VLA-Checkpoint-config.json",
            "path/to/SimplerEnv-eval-logs"
        )
    ]
    
    # 要执行的评估脚本列表（固定顺序）
    task_scripts = [
        "openvla_pick_coke_can_visual_matching.sh",
        "openvla_move_near_visual_matching.sh",
        "openvla_put_in_drawer_visual_matching.sh",
        "openvla_drawer_visual_matching.sh"
    ]
    
    # ===================== 安全验证 =====================
    # 1. 验证GPU数量是否足够
    if len(task_scripts) > num_gpus:
        print(f"❌ 错误: 任务数量({len(task_scripts)})超过可用GPU数量({num_gpus})")
        sys.exit(1)
    
    # 2. 验证所有脚本是否存在
    scripts_dir = "scripts"
    for script in task_scripts:
        script_path = os.path.join(scripts_dir, script)
        if not os.path.exists(script_path):
            print(f"❌ 错误: 脚本文件不存在 - {script_path}")
            sys.exit(1)
    
    # ===================== 主执行逻辑 =====================
    for idx, (ckpt, config, log_path) in enumerate(ckpt_paths):
        print(f"\n{'='*50}")
        print(f"开始评估检查点 #{idx+1}/{len(ckpt_paths)}")
        print(f"模型路径: {ckpt}")
        print(f"配置文件: {config}")
        print(f"日志目录: {log_path}")
        print(f"{'='*50}")
        
        # 1. 创建日志目录（带错误处理）
        try:
            os.makedirs(log_path, exist_ok=True)
            print(f"✅ 创建/确认日志目录: {log_path}")
        except Exception as e:
            print(f"❌ 创建日志目录失败: {e}")
            continue  # 跳过当前检查点
        
        # 2. 验证检查点和配置文件是否存在
        if not os.path.exists(ckpt):
            print(f"❌ 检查点文件不存在: {ckpt}")
            continue
        
        if not os.path.exists(config):
            print(f"❌ 配置文件不存在: {config}")
            continue
        
        # 3. 创建任务监控器
        task_monitor = TaskMonitor()
        
        # 4. 启动所有任务（每个任务在专用GPU上）
        for task_idx, script_name in enumerate(task_scripts):
            gpu_id = task_idx  # 每个任务分配到单独的GPU (0-3)
            task_name = script_name.replace('.sh', '')
            task_id = f"{task_name}_{gpu_id}"
            
            # 构建日志文件路径
            log_file = os.path.join(log_path, f"{task_name}_gpu{gpu_id}.log")
            
            # 构建执行命令
            bash_command = (
                f"bash scripts/{script_name} "
                f"{ckpt} {config}"
            )
            
            # 构建nohup命令
            nohup_cmd = (
                f"nohup env CUDA_VISIBLE_DEVICES={gpu_id} {bash_command} "
                f"> {log_file} 2>&1"
            )
            
            # 使用subprocess.Popen启动进程并获取PID
            try:
                process = subprocess.Popen(
                    nohup_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True  # 创建新会话组
                )
                
                # 记录任务信息
                task_info = {
                    "script": script_name,
                    "gpu": gpu_id,
                    "log": log_file,
                    "pid": process.pid,
                    "process": process,
                    "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "command": nohup_cmd
                }
                
                # 添加到监控器
                task_monitor.add_task(task_id, task_info)
                
                print(f"✅ 成功启动任务 {task_idx+1}/{len(task_scripts)}: "
                      f"{script_name} (GPU {gpu_id})")
                print(f"   ├─ PID: {process.pid}")
                print(f"   ├─ 日志文件: {log_file}")
                print(f"   └─ 启动时间: {task_info['start_time']}")
                
            except Exception as e:
                print(f"❌ 启动任务失败: {script_name} (GPU {gpu_id})")
                print(f"   错误信息: {e}")
                continue
        
        # 5. 监控任务状态 - 独立线程
        def monitor_tasks(monitor):
            print("\n⏳ 启动任务监控线程...")
            print("任务状态将每分钟更新一次")
            
            # 持续监控直到所有任务完成
            while not monitor.all_completed():
                active_tasks = monitor.get_active_tasks()
                print(f"\n当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"活动任务: {len(active_tasks)}/{len(task_scripts)}")
                
                for task_id in active_tasks:
                    task_info = monitor.get_task_info(task_id)
                    
                    # 检查进程状态
                    status = task_info["process"].poll()
                    if status is None:
                        # 进程仍在运行
                        print(f"  - {task_id}: 运行中 (PID: {task_info['pid']})")
                    else:
                        # 进程已完成
                        monitor.mark_completed(task_id)
                        print(f"  - {task_id}: 已完成 (退出码: {status})")
                        
                        # 记录完成时间
                        task_info["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        task_info["exit_code"] = status
                
                # 如果还有任务在运行，等待一分钟
                if not monitor.all_completed():
                    time.sleep(60)
            
            print("\n✅ 所有任务已完成!")
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_tasks, args=(task_monitor,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 6. 主线程等待所有任务完成
        print("\n主线程等待所有任务完成...")
        try:
            # 等待监控线程结束（即所有任务完成）
            monitor_thread.join()
            
            # 最终状态报告
            print("\n" + "="*60)
            print("任务完成报告:")
            print("="*60)
            
            for task_id in task_monitor.active_tasks:
                task_info = task_monitor.get_task_info(task_id)
                status = "已完成" if task_id in task_monitor.completed_tasks else "未知状态"
                
                print(f"\n任务: {task_info['script']} (GPU {task_info['gpu']})")
                print(f"  PID: {task_info['pid']}")
                print(f"  状态: {status}")
                print(f"  开始时间: {task_info['start_time']}")
                
                if "end_time" in task_info:
                    print(f"  结束时间: {task_info['end_time']}")
                    print(f"  退出码: {task_info.get('exit_code', 'N/A')}")
                
                print(f"  日志文件: {task_info['log']}")
                
                # 检查日志是否存在
                if os.path.exists(task_info['log']):
                    # 检查日志中的错误
                    with open(task_info['log'], 'r') as f:
                        content = f.read()
                        if "error" in content.lower() or "exception" in content.lower():
                            print("  ⚠️ 警告: 日志中包含错误信息!")
                        else:
                            print("  日志状态: 正常")
                else:
                    print("  ⚠️ 警告: 日志文件未找到!")
            
            print("\n" + "="*60)
        
        except KeyboardInterrupt:
            print("\n⚠️ 检测到Ctrl+C，准备退出...")
            
            # 终止所有仍在运行的任务
            print("终止仍在运行的任务...")
            for task_id in task_monitor.get_active_tasks():
                task_info = task_monitor.get_task_info(task_id)
                if task_info["process"].poll() is None:
                    print(f"  终止任务: {task_id} (PID: {task_info['pid']})")
                    try:
                        # 终止整个进程组
                        os.killpg(os.getpgid(task_info["pid"]), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            
            print("已发送终止信号，等待进程退出...")
            time.sleep(5)
            sys.exit(1)

if __name__ == "__main__":
    print("="*60)
    print("多GPU评估任务启动器")
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        run_parallel_evaluations()
    except Exception as e:
        print(f"❌ 发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("所有评估任务处理完成!")
    print(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)