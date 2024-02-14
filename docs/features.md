# Features
This page provides a description of individual data features in the datasets. Features available in each dataset are listed on the [dataset metadata][metadata] page.

## PPI sequence
A per-packet information (PPI) sequence is a 2D matrix describing the first 30 packets of a flow. For flows shorter than 30 packets, the PPI sequence is padded with zeros.
When data scalers (`psizes_scaler` and `ipt_scaler` config options) are enabled, this padding is ignored during the scaling transformation, and padding zeroes are kept in the final scaled data.
The `zero_ppi_start` config option can be used to zero out the first *N* packets of PPI sequences, which is useful when evaluating the importance of handshake packets that are transmitted at the start of each connection.

| **Name**                | **Description**                                                                     | **Config options**                 |
|-------------------------|-------------------------------------------------------------------------------------|------------------------------------|
| SIZE                    | Size of the transport payload                                                       | `psizes_scaler`, `psizes_max`      |
| IPT                     | Inter-packet time in milliseconds. The IPT of the first packet is set to zero       | `ipt_scaler`, `ipt_min`, `ipt_max` |
| DIR                     | Direction of the packet encoded as ±1                                               |                                    |
| PUSH_FLAG               | Whether the push flag was set in the TCP packet                                     | `use_push_flags`                   |

## Flow statistics
Flow statistics are standard features describing the entire flow (with exceptions of *PPI_* features that relate to the PPI sequence of the given flow). *_REV* features correspond to the reverse (server to client) direction.

| **Name**                | **Description**                                                                     | **Config options**                   |
|-------------------------|-------------------------------------------------------------------------------------|--------------------------------------|
| DURATION                | Duration of the flow in seconds                                                     | `flowstats_scaler`                   |
| BYTES                   | Number of transmitted bytes from client to server                                   | `flowstats_scaler`, `flowstats_clip` |
| BYTES_REV               | Number of transmitted bytes from server to client                                   | `flowstats_scaler`, `flowstats_clip` |
| PACKETS                 | Number of packets transmitted from client to server                                 | `flowstats_scaler`, `flowstats_clip` |
| PACKETS_REV             | Number of packets transmitted from server to client                                 | `flowstats_scaler`, `flowstats_clip` |
| PPI_LEN                 | Number of packets in the PPI sequence                                               | `flowstats_scaler`                   |
| PPI_DURATION            | Duration of the PPI sequence in seconds                                             | `flowstats_scaler`                   |
| PPI_ROUNDTRIPS          | Number of roundtrips in the PPI sequence                                            | `flowstats_scaler`                   |
| FLOW_ENDREASON_IDLE     | Flow was terminated because it was idle                                             |                                      |
| FLOW_ENDREASON_ACTIVE   | Flow was terminated because it reached the active timeout                           |                                      |
| FLOW_ENDREASON_OTHER    | Flow was terminated for other reasons                                               |                                      |

## Packet histograms
Packet histograms include binned counts of packet sizes and inter-packet times of the entire flow.
There are 8 bins with a logarithmic scale; the intervals are 0–15, 16–31, 32–63, 64–127, 128–255, 256–511, 512–1024, >1024 [ms or B]. The units are milliseconds for inter-packet times and bytes for packet sizes.
The histograms are built from all packets of the entire flow, unlike PPI sequences that describe the first 30 packets.

| **Name**                | **Description**                                                                     |  **Config options**                                    |
|-------------------------|-------------------------------------------------------------------------------------|--------------------------------------------------------|
| PSIZE_BIN{*x*}          | Packet sizes histogram *x*-th bin for the forward direction                         | `use_packet_histograms`, `normalize_packet_histograms` |
| PSIZE_BIN{*x*}_REV      | Packet sizes histogram *x*-th bin for the reverse direction                         | `use_packet_histograms`, `normalize_packet_histograms` |
| IPT_BIN{*x*}            | Inter-packet times histogram *x*-th bin for the forward direction                   | `use_packet_histograms`, `normalize_packet_histograms` |
| IPT_BIN{*x*}_REV        | Inter-packet times histogram *x*-th bin for the reverse direction                   | `use_packet_histograms`, `normalize_packet_histograms` |

On the [dataset metadata][metadata] page, packet histogram features are called `PHIST_SRC_SIZES`, `PHIST_DST_SIZES`, `PHIST_SRC_IPT`, `PHIST_DST_IPT`. Those are the names of database columns that are flattened to the _BIN{*x*} features.

## TCP features
Datasets with TLS over TCP traffic contain features indicating the presence of individual TCP flags in the flow. A subset of flags defined in `cesnet_datazoo.constants.SELECTED_TCP_FLAGS` is used.

| **Name**         | **Description**                                                                            |  **Config options**             |
|------------------|--------------------------------------------------------------------------------------------|                                 |
| FLAG_{*F*}       | Whether *F* flag was present in the forward (client to server) direction                   | `use_tcp_features`              |
| FLAG_{*F*}_REV   | Whether *F* flag was present in the reverse (server to client) direction                   | `use_tcp_features`              |

## Other fields
Datasets contain auxiliary information about samples, such as communicating hosts, flow times, and more fields extracted from the ClientHello message. The [dataset metadata][metadata] page lists available fields in individual datasets. 

| **Name**                | **Description**                                                                     | **Config options**              |
|-------------------------|-------------------------------------------------------------------------------------|                                 |
| ID                      | Per-dataset unique flow identifier                                                  | `return_other_fields`           |
| TIME_FIRST              | Timestamp of the first packet                                                       | `return_other_fields`           |
| TIME_LAST               | Timestamp of the last packet                                                        | `return_other_fields`           |
| SRC_IP                  | Source IP address                                                                   | `return_other_fields`           |
| DST_IP                  | Destination IP address                                                              | `return_other_fields`           |
| DST_ASN                 | Destination Autonomous System number                                                | `return_other_fields`           |
| SRC_PORT                | Source port                                                                         | `return_other_fields`           |
| DST_PORT                | Destination port                                                                    | `return_other_fields`           |
| PROTOCOL                | Transport protocol                                                                  | `return_other_fields`           |
| TLS_SNI / QUIC_SNI      | Server Name Indication domain                                                       | `return_other_fields`           |
| TLS_JA3                 | JA3 fingerprint                                                                     | `return_other_fields`           |
| QUIC_VERSION            | QUIC protocol version                                                               | `return_other_fields`           |
| QUIC_USER_AGENT         | User agent string if available in the QUIC Initial Packet                           | `return_other_fields`           |
<!-- 
| APP                     | Web service label                                                                   |                                 |
| CATEGORY                | Service category label                                                              |                                 | 
-->

## Details about packet histograms and PPI
Due to differences in implementation between packet sequences ([pstats.cpp](https://github.com/CESNET/ipfixprobe/blob/master/process/pstats.cpp)) and packet histogram ([phist.cpp](https://github.com/CESNET/ipfixprobe/blob/master/process/phists.cpp)) plugins of the ipfixprobe exporter, the number of packets in PPI and histograms can differ (even for flows shorter than 30 packets). The differences are summarized in the following table.
Note that this is related to TLS over TCP datasets.

| *TLS over TCP datasets*                                       | Packet histograms | PPI sequence     | PACKETS and PACKET_REV |
|---------------------------------------------------------------|-------------------|------------------|------------------------|
| **Zero-length packets**<br>(without L4 payload, e.g. ACKs)    | Not included      | Not included     | Included               |
| **Retransmissions**<br>(and out-of-order packets)             | Included          | Not included*\** | Included               |
| **Computed from**                                             | Entire flow       | First 30 packets | Entire flow            |

**The implementation for the detection of TCP retransmissions and out-of-order packets is far from perfect. Packets with a non-increasing SEQ number are skipped.*

For QUIC, there is no detection of retransmissions or out-of-order packets, and QUIC acknowledgment packets are included in both packet sequences and packet histograms.