"""Render the two historical figures from the shipped CSV files; no device access."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK SC', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#777777', 'text.color': '#202a35',
                     'axes.labelcolor': '#202a35', 'xtick.color': '#4b5563', 'ytick.color': '#4b5563'})
BLUE, GOLD = '#28689b', '#af7714'


def rows(name):
    with (ROOT / 'data' / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    validation = rows('historical-validation.csv')
    fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
    x = [int(r['step']) // 1000 for r in validation]
    y = [float(r['episode_macro_normalized_l1']) for r in validation]
    reference = float(validation[0]['hold_reference'])
    ax.plot(x, y, '-o', color=BLUE, lw=2.3, markersize=6, label='ACT · 冻结验证集')
    ax.axhline(reference, color=GOLD, lw=1.8, ls='--', label=f'保持 state 参照 · {reference:.5f}')
    for index, offset in [(0, (0, 10)), (4, (0, -22)), (7, (0, -22))]:
        ax.annotate(f'{y[index]:.5f}', (x[index], y[index]), xytext=offset,
                    textcoords='offset points', ha='center', fontsize=11)
    ax.set(xticks=x, ylim=(0, .095), xlabel='训练更新步数（千步）', ylabel='归一化 L1 · episode 平均',
           title='历史外部单相机：训练步数与离线验证误差\n2026-09-22–23 · 同一 10 条示范 / 17,433 帧')
    ax.grid(axis='y', color='#e4e7eb', lw=.8)
    ax.legend(loc='lower left', frameon=False)
    fig.savefig(ROOT / 'assets/historical-validation.png', dpi=160)
    plt.close(fig)

    data = rows('historical-gripper.csv')
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True, layout='constrained')
    for number, ax in enumerate(axes, 1):
        trial = [r for r in data if int(r['trial']) == number]
        times = [float(r['time_seconds']) for r in trial]
        target = [float(r['te_target']) for r in trial]
        state = [float(r['measured_state']) for r in trial]
        ax.plot(times, target, color=BLUE, lw=1.8, label='TE 目标')
        ax.plot(times, state, color=GOLD, lw=1.6, ls='--', label='实测反馈')
        ax.set(xlim=(0, 60), ylim=(-2, 24), ylabel='夹爪 · 标定单位',
               title=f'试验 {number} · 未抓起 · 1,787 个发送动作')
        peak = max(state)
        peak_i = state.index(peak)
        ax.annotate(f'反馈峰值 {peak:.3f}', (times[peak_i], peak), xytext=(20, 12),
                    textcoords='offset points', fontsize=11, arrowprops={'arrowstyle': '-', 'color': '#777777'})
        ax.grid(axis='y', color='#e4e7eb', lw=.8)
        ax.legend(loc='upper left', frameon=False)
    axes[-1].set_xlabel('动作日志时间（秒）')
    fig.suptitle('2026-10-10 · 旧单相机两次抓取的夹爪轨迹\n同一 100K 模型 · TE=0.01 · chunk=16 · 每次执行 1 步', fontsize=15)
    fig.savefig(ROOT / 'assets/historical-gripper.png', dpi=160)
    plt.close(fig)
    print('Rendered 2 historical PNG figures.')


if __name__ == '__main__':
    main()
