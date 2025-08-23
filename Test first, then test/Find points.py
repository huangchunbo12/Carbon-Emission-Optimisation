import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

# 设置新罗马字体
plt.rcParams['font.family'] = 'Times New Roman'

# 定义积分上限函数
def upper_limit_function(delta):
    return np.exp(-1 / (4 * np.arctan(delta)))

# 定义delta的范围
delta_values = np.linspace(0, 1, 500)
upper_limits = upper_limit_function(delta_values)

# 计算积分结果，先对r进行积分，然后对delta进行积分
def integrand_4(delta):
    upper_limit = upper_limit_function(delta)
    return upper_limit  # 积分结果为上限，因r的积分是1

# 进行数值积分
result_4, error_4 = quad(integrand_4, 0, 1)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(delta_values, upper_limits, label=r'$e^{-\frac{1}{4} \arctan(\delta)}$', color='b')
plt.fill_between(delta_values, 0, upper_limits, alpha=0.3, color='b', label='Area under curve')

# 添加标题和标签
plt.title('Visualization of the Integral Region')
plt.xlabel(r'$\delta$')
plt.ylabel(r'$e^{-\frac{1}{4} \arctan(\delta)}$')
plt.legend()
plt.grid(True)
plt.show()

# 输出积分结果和误差
print(f"积分结果: {result_4:.4f}")
print(f"误差估计: {error_4:.4e}")
