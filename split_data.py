import logging
from config import config as cfg
from utilities.data_wrangling import load_data, split_dataset, concatenate_data, export_data, preprocessing

logger = logging.getLogger(cfg.logger_app_name)


def main():
    """

    :return:
    """
    logger.info('El proceso ha comenzado... \n')
    df = load_data(filename=f'{cfg.DATA_FOLDER}/{cfg.INPUT_FILE}', sep=';',
                   decimal=',', dtype=cfg.FEATURES_DTYPES.update(cfg.LABEL_DTYPE))
    df = preprocessing(df=df)
    train_data, validation_data, test_data, train_label, validation_label, test_label = split_dataset(df=df,
                                                                                                      label=cfg.LABEL,
                                                                                                      stratify=False)
    train_df = concatenate_data(train_data, train_label)
    validation_df = concatenate_data(validation_data, validation_label)
    export_data(df=train_df, file_path=f'{cfg.DATA_FOLDER}/train.csv', with_header=True)
    export_data(df=validation_df, file_path=f'{cfg.DATA_FOLDER}/validation.csv', with_header=True)
    export_data(df=test_data, file_path=f'{cfg.DATA_FOLDER}/test.csv', with_header=True)
    export_data(df=test_label, file_path=f'{cfg.DATA_FOLDER}/test_label.csv', with_header=True)
    logger.info('El proceso ha finalizado correctamente.')
    return None


if __name__ == '__main__':
    main()
