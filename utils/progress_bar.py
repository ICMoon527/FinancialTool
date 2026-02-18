from tqdm import tqdm


class ProgressBar:
    def __init__(self, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, 
                 length: int = 30, fill: str = '█'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.current = 0
        self.is_completed = False
        # 使用tqdm创建进度条
        self.tqdm_bar = tqdm(
            total=total,
            desc=prefix,
            unit='item',
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
    
    def update(self, current: int = None):
        if self.is_completed:
            return
        
        if current is not None:
            # 更新到指定位置
            self.tqdm_bar.n = current
            self.current = current
        else:
            # 默认更新1
            self.tqdm_bar.update(1)
            self.current += 1
        
        self.tqdm_bar.refresh()
        
        if self.current >= self.total:
            self.is_completed = True
            self.tqdm_bar.close()
    
    def finish(self):
        """确保进度条完成并添加换行"""
        if not self.is_completed:
            self.current = self.total
            self.update(current=self.total)
        else:
            self.tqdm_bar.close()
        print(self.suffix)


def create_progress_bar(total: int, description: str = '处理中'):
    return ProgressBar(total=total, prefix=f'{description}', suffix='完成')
