import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取 Excel 数据
#    将路径改成你自己的文件位置，例如 r"C:\Users\xxx\Documents\排放数据.xlsx"
file_path = r"C:\Users\工作簿1.xlsx"
df = pd.read_excel(file_path)

plt.rcParams["font.sans-serif"] = ["SimHei"]          # 指定中文黑体
plt.rcParams["axes.unicode_minus"] = False            # 解决负号显示问题

# 2. 识别年份列与省份列（默认第一列是“年份”，其余为省份）
year_col = df.columns[0]
province_cols = df.columns[1:]

# 3. 找出 2022 年排放量最大的前五省份
last_year = df[year_col].max()                        # 最大年份（如 2022）
top5_series = (
    df.loc[df[year_col] == last_year, province_cols]  # 取该年各省排放
      .T.squeeze()                                    # 转置成 Series
      .sort_values(ascending=False)                   # 从高到低排序
      .head(5)                                        # 前五
)
top5 = top5_series.index.tolist()

# 4. 开始绘图
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(top5)))   # 为前五省份准备配色

for col in province_cols:
    if col in top5:
        # 前五省份：彩色、粗线
        plt.plot(
            df[year_col], df[col],
            label=f"{col}（{int(top5_series[col]):,} 吨）",
            linewidth=2.5,
            color=colors[top5.index(col)]
        )
    else:
        # 其他省份：灰色、细线、透明度低
        plt.plot(
            df[year_col], df[col],
            color="grey",
            linewidth=0.8,
            alpha=0.35
        )

# 5. 美化图表
plt.title("2000–2022 年各省碳排放趋势（突出 2022 年排放前五）", fontsize=15)
plt.xlabel("年份", fontsize=13)
plt.ylabel("碳排放量（吨）", fontsize=13)

# 放大坐标轴刻度文字
plt.tick_params(axis="both", labelsize=13)            # 统一 x、y 轴
# 也可分开：plt.xticks(fontsize=13); plt.yticks(fontsize=13)

plt.grid(linestyle="--", linewidth=0.5, alpha=0.6)
plt.legend(
    title="Top 5 省份（2022 年排放量）",
    fontsize="small",
    title_fontsize="small",
    loc="upper left"
)
plt.tight_layout()
plt.show()
