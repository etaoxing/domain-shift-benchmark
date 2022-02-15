from dsb.dependencies import *

import pandas as pd
import scipy.stats


color2num = dict(
    gray=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37, crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class TimestampLogger:
    def __init__(self, timestamp_color='yellow', log_path=None, mode='a'):
        self.out = sys.stdout
        self.timestamp_color = timestamp_color
        self.nl = True  # newline
        if log_path is not None:
            self.log = open(log_path, mode)
        else:
            self.log = None

    def flush(self):
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.out, attr)

    def _write(self, x):
        self.out.write(x)
        if self.log is not None:
            self.log.write(x)
            self.log.flush()

    def write(self, x):
        if x == '\n' or x == '':
            self._write(x)
            self.nl = True
        elif self.nl:
            timestamp = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            timestamp = colorize(timestamp, self.timestamp_color)
            sep = '\n' if len(x) >= 80 else ' '
            self._write(f"[{timestamp}]{sep}{x}")
            self.nl = False
        else:
            self._write(x)


class TabularLogger:
    def __init__(self, filename, use_tensorboard=True):
        self.filename = filename
        self.df = pd.DataFrame()
        self.num_cols = 0
        self.it_index_label = '_diagnostics/total_iterations'
        if use_tensorboard:
            tb_path = '/'.join(self.filename.split('/')[:-1])
            filename_suffix = self.filename.split('/')[-1].split('.')[0]

            import warnings

            warnings.filterwarnings('ignore')
            # import this here to hide this warning w/ multiprocessing:
            # WARNING:root:This caffe2 python run does not have GPU support. Will run in CPU only mode.
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(tb_path, filename_suffix=filename_suffix)
        else:
            self.tb_writer = None

    def log(self, it):
        row = self.df.loc[self.df[self.it_index_label] == it].iloc[0]
        row = row[pd.notna(row)].sort_index()
        print(row.to_string(float_format=lambda x: f"{x:.6f}"))
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
            except:
                pass

    def record(self, data):
        iterations = data.pop('iteration')
        self.cur_it = iterations[-1]
        self.cur_row = {self.it_index_label: self.cur_it}

        for k, v in data.items():
            try:
                if k[0] == '_':
                    self.record_tabular(k, v)
                else:
                    if isinstance(v, np.ndarray) or isinstance(v, list) or isinstance(v, tuple):
                        v = list(filter(lambda x: x is not None, v))
                        mean_std_only = k.split('/')[-1][0] == '_'
                        self.record_tabular_misc_stat(k, v, mean_std_only=mean_std_only)
                    else:
                        # assume scalar
                        self.record_tabular(k, v)
            except Exception as e:
                # NOTE: if operands have different shape, then last batch size is prob different
                print(f'ERROR: cannot log key: {k}')
                raise e

        self.df = self.df.append(self.cur_row, ignore_index=True)

        if not os.path.exists(self.filename) or len(self.df.columns) != self.num_cols:
            self.df.to_csv(self.filename, index=False)
        else:
            self.df.loc[self.df[self.it_index_label] == self.cur_it].to_csv(
                self.filename, header=False, index=False, mode='a'
            )

        self.num_cols = len(self.df.columns)

    def record_tabular_misc_stat(self, key, values, mean_std_only=False):
        prefix = key + "/"
        suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Mean" + suffix, np.mean(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values, ddof=1))

            if not mean_std_only:
                self.record_tabular(prefix + "Median" + suffix, np.median(values))
                self.record_tabular(prefix + "Min" + suffix, np.min(values))
                self.record_tabular(prefix + "Max" + suffix, np.max(values))

                # https://github.com/google-research/rliable/blob/e064a52acd3fec467e91a7607b7ce014aa9f38ca/rliable/metrics.py#L61
                self.record_tabular(
                    prefix + "MeanIQ" + suffix, scipy.stats.trim_mean(values, proportiontocut=0.25)
                )
                self.record_tabular(prefix + "SEM" + suffix, scipy.stats.sem(values, ddof=1))

                # iqr = scipy.stats.iqr(values)
                # self.record_tabular(prefix + "IQR25" + suffix, iqr[0])
                # self.record_tabular(prefix + "IQR75" + suffix, iqr[1])

    def record_tabular(self, k, v):
        self.cur_row[k] = v
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(k, v, global_step=self.cur_it)

    def record_cfg(self, cfg):
        if self.tb_writer is not None:
            d = pd.DataFrame.from_dict(cfg, orient='index')
            self.tb_writer.add_text('cfg', d.to_markdown(), 0)

    def record_argparse(self, args, extras):
        command = str(args) + ' ' + str(extras)

        if self.tb_writer is not None:
            self.tb_writer.add_text('command', command, 0)
