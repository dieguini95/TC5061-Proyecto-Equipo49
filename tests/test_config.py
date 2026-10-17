import pandas as pd

from mibici_ml import config


def test_particion_temporal_ordenada_y_sin_traslape():
    train = (pd.Timestamp(config.TRAIN_START), pd.Timestamp(config.TRAIN_END))
    val = (pd.Timestamp(config.VAL_START), pd.Timestamp(config.VAL_END))
    test = (pd.Timestamp(config.TEST_START), pd.Timestamp(config.TEST_END))

    for inicio, fin in (train, val, test):
        assert inicio <= fin
    assert train[1] < val[0]
    assert val[1] < test[0]


def test_ventana_de_datos_cubre_la_particion():
    inicio_datos = pd.Timestamp(config.START_MONTH + "-01")
    fin_datos = pd.Timestamp(config.END_MONTH + "-01") + pd.offsets.MonthEnd(0)
    assert inicio_datos <= pd.Timestamp(config.TRAIN_START)
    assert pd.Timestamp(config.TEST_END) <= fin_datos


def test_horizontes_y_purga_son_positivos():
    assert config.P2_HORIZON_HOURS > 0
    assert config.P1_PURGE_GAP_HOURS >= 24
    assert all(lag >= 24 for lag in config.P1_LAG_HOURS)
    assert 0 < config.P2_TOP_K_FRACTION <= 1
