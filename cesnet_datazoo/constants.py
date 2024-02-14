DATASET_SIZES = {
    "XS": 10_000_000,
    "S": 25_000_000,
    "M": 50_000_000,
    "L": 100_000_000,
}

# Per-packet information (PPI) constants
IPT_POS = 0
DIR_POS = 1
SIZE_POS = 2
PUSH_FLAGS_POS = 3
PPI_MAX_LEN = 30
TCP_PPI_CHANNELS = 4
UDP_PPI_CHANNELS = 3

# Features
FLOWSTATS_TO_SCALE =  ["BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "PPI_LEN", "PPI_ROUNDTRIPS", "PPI_DURATION", "DURATION"]
FLOWSTATS_NO_CLIP = ["PPI_LEN", "PPI_ROUNDTRIPS", "PPI_DURATION", "DURATION"]
SELECTED_TCP_FLAGS = ["FLAG_CWR", "FLAG_CWR_REV", "FLAG_ECE", "FLAG_ECE_REV", "FLAG_PSH_REV", "FLAG_RST", "FLAG_RST_REV", "FLAG_FIN", "FLAG_FIN_REV"]
FLOWEND_REASON_FEATURES = ["FLOW_ENDREASON_IDLE", "FLOW_ENDREASON_ACTIVE", "FLOW_ENDREASON_END", "FLOW_ENDREASON_OTHER"]
PHISTS_FEATURES = ["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"]
PHIST_BIN_COUNT = 8

# Column names
APP_COLUMN = "APP"
CATEGORY_COLUMN = "CATEGORY"
PPI_COLUMN = "PPI"
TLS_SNI_COLUMN = "TLS_SNI"
QUIC_SNI_COLUMN = "QUIC_SNI"

# Servicemap constants
SERVICEMAP_PROVIDER_COLUMN = "Service Provider"
SERVICEMAP_CATEGORY_COLUMN = "Service Category"
SERVICEMAP_FILE = "servicemap.csv"

# Class labels
UNKNOWN_STR_LABEL = "_unknown"
DEFAULT_BACKGROUND_CLASS = "default-background"
GOOGLE_BACKGROUND_CLASS = "google-background"

# Indices
INDICES_TABLE_POS = 0
INDICES_INDEX_POS = 1
INDICES_LABEL_POS = 2
