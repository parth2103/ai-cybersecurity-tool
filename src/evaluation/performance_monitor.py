import time
import psutil
import numpy as np
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        
    def measure_inference_time(self, model, X_batch):
        """Measure model inference time"""
        start_time = time.time()
        _ = model.predict(X_batch)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        self.inference_times.append(inference_time)
        
        return inference_time
    
    def get_system_metrics(self):
        """Get current system metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_mb': psutil.virtual_memory().used / (1024 * 1024)
        }
        
        self.cpu_usage.append(metrics['cpu_percent'])
        self.memory_usage.append(metrics['memory_percent'])
        
        return metrics
    
    def get_performance_summary(self):
        """Get summary statistics"""
        if not self.inference_times:
            return None
            
        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'p95_inference_time_ms': np.percentile(self.inference_times, 95),
            'p99_inference_time_ms': np.percentile(self.inference_times, 99),
            'avg_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'avg_memory_percent': np.mean(self.memory_usage) if self.memory_usage else 0,
            'throughput_per_second': 1000 / np.mean(self.inference_times) if self.inference_times else 0
        }
    
    def check_performance_requirements(self, max_latency_ms=5000):
        """Check if performance meets requirements"""
        if not self.inference_times:
            return True
            
        avg_time = np.mean(self.inference_times)
        meets_requirement = avg_time < max_latency_ms
        
        if not meets_requirement:
            print(f"⚠️ Performance Warning: Average inference time ({avg_time:.2f}ms) exceeds requirement ({max_latency_ms}ms)")
        else:
            print(f"✅ Performance OK: Average inference time ({avg_time:.2f}ms) within requirement")
            
        return meets_requirement
