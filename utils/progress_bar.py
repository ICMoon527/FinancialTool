import sys


class ProgressBar:
    def __init__(self, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, 
                 length: int = 50, fill: str = '█'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.current = 0
    
    def update(self, current: int = None):
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.current / float(self.total)))
        filled_length = int(self.length * self.current // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}')
        sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write('\n')


def create_progress_bar(total: int, description: str = '处理中'):
    return ProgressBar(total=total, prefix=f'{description}:', suffix='完成')
