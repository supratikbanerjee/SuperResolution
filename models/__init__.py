def create_model(config, logger):
    model = config['model']

    if model == 'cnn':
        from .cnn_trainer import CNN as M
    elif model == 'gan':
        from .gan_trainer import GAN as M
    elif model == 'prosr':
        pass
        #from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(config)
    logger.log('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
