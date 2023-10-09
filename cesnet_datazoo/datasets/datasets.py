from cesnet_datazoo.datasets.cesnet_dataset import CesnetDataset


class CESNET_TLS22(CesnetDataset):
    """Dataset class for [CESNET-TLS22][cesnet-tls22]."""
    name = "CESNET-TLS22"
    database_filename = "CESNET-TLS22.h5"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=cesnet-tls22"
    time_periods = {
        "W-2021-40": ["20211004", "20211005", "20211006", "20211007", "20211008", "20211009", "20211010"],
        "W-2021-41": ["20211011", "20211012", "20211013", "20211014", "20211015", "20211016", "20211017"],
    }
    default_train_period = "W-2021-40"
    default_test_period = "W-2021-41"

class CESNET_QUIC22(CesnetDataset):
    """Dataset class for [CESNET-QUIC22][cesnet-quic22]."""
    name = "CESNET-QUIC22"
    database_filename = "CESNET-QUIC22.h5"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=cesnet-quic22"
    time_periods = {
        "W-2022-44": ["20221031", "20221101", "20221102", "20221103", "20221104", "20221105", "20221106"],
        "W-2022-45": ["20221107", "20221108", "20221109", "20221110", "20221111", "20221112", "20221113"],
        "W-2022-46": ["20221114", "20221115", "20221116", "20221117", "20221118", "20221119", "20221120"],
        "W-2022-47": ["20221121", "20221122", "20221123", "20221124", "20221125", "20221126", "20221127"],
        "W45-47": ["20221107", "20221108", "20221109", "20221110", "20221111", "20221112", "20221113",
                        "20221114", "20221115", "20221116", "20221117", "20221118", "20221119", "20221120",
                        "20221121", "20221122", "20221123", "20221124", "20221125", "20221126", "20221127"],
    }
    default_train_period = "W-2022-44"
    default_test_period = "W-2022-45"

class CESNET_TLS_Year22(CesnetDataset):
    """Dataset class for [CESNET-TLS-Year22][cesnet-tls-year22]."""
    name = "CESNET-TLS-Year22"
    database_filename = "CESNET-TLS-Year22.h5"
    bucket_url = "https://liberouter.org/datazoo/download?bucket=cesnet-tls-year22"
    time_periods = {f"W-2022-{week}": [] for week in range(1, 53)} | {f"M-2022-{month}": [] for month in range(1, 13)}
    time_periods_gen = True
    default_train_period = "M-2022-9"
    default_test_period = "M-2022-10"
