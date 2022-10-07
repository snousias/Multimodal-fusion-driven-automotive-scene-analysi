from utils.config import Config
from utils.misc import init_env


cfg = Config().parse()
init_env(cfg)

if cfg.mode == 'train':
    from train import train
    train(cfg)
elif cfg.mode == 'eval':
    from eval import eval
    eval(cfg)
elif cfg.mode == 'sharing':
    from train_weight_sharing import train_weight_sharing
    train_weight_sharing(cfg)
elif cfg.mode == 'demo':
    from demo import demo
    demo(cfg)
else:
    raise ValueError('Mode {} is invalid.'.format(cfg.mode))
