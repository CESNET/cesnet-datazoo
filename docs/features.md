# Features
This page provides a description of individual data features in the datasets. Features available in each dataset are listed on the [dataset metadata][metadata] page.

## PPI sequence
A per-packet information (PPI) sequence is a 2D matrix describing the first 30 packets of a flow. For flows shorter than 30 packets, the PPI sequence is padded with zeros.
Set `use_push_flags` to include PUSH flags in PPI sequences, if available in the dataset.

| **Name**                | **Description**                                                                     |
|-------------------------|-------------------------------------------------------------------------------------|
| SIZE                    | Size of the transport payload                                                       |
| IPT                     | Inter-packet time in milliseconds. The IPT of the first packet is set to zero       |
| DIR                     | Direction of the packet encoded as ±1                                               |
| PUSH_FLAG               | Whether the push flag was set in the TCP packet                                     |

## Flow statistics
Flow statistics are standard features describing the entire flow (with exceptions of *PPI_* features that relate to the PPI sequence of the given flow). *_REV* features correspond to the reverse (server to client) direction.

| **Name**                | **Description**                                                                     |
|-------------------------|-------------------------------------------------------------------------------------|
| DURATION                | Duration of the flow in seconds                                                     |
| BYTES                   | Number of transmitted bytes from client to server                                   |
| BYTES_REV               | Number of transmitted bytes from server to client                                   |
| PACKETS                 | Number of packets transmitted from client to server                                 |
| PACKETS_REV             | Number of packets transmitted from server to client                                 |
| PPI_LEN                 | Number of packets in the PPI sequence                                               |
| PPI_DURATION            | Duration of the PPI sequence in seconds                                             |
| PPI_ROUNDTRIPS          | Number of roundtrips in the PPI sequence                                            |
| FLOW_ENDREASON_IDLE     | Flow was terminated because it was idle                                             |
| FLOW_ENDREASON_ACTIVE   | Flow was terminated because it reached the active timeout                           |
| FLOW_ENDREASON_OTHER    | Flow was terminated for other reasons                                               |

## Packet histograms
Packet histograms include binned counts of packet sizes and inter-packet times of the entire flow.
There are 8 bins with a logarithmic scale; the intervals are 0–15, 16–31, 32–63, 64–127, 128–255, 256–511, 512–1024, >1024 [ms or B]. The units are milliseconds for inter-packet times and bytes for packet sizes.
The histograms are built from all packets of the entire flow, unlike PPI sequences that describe the first 30 packets.
Set `use_packet_histograms` for using packet histograms features, if available in the dataset.

| **Name**                | **Description**                                                                     |
|-------------------------|-------------------------------------------------------------------------------------|
| PSIZE_BIN{*x*}          | Packet sizes histogram *x*-th bin for the forward direction                         |
| PSIZE_BIN{*x*}_REV      | Packet sizes histogram *x*-th bin for the reverse direction                         |
| IPT_BIN{*x*}            | Inter-packet times histogram *x*-th bin for the forward direction                   |
| IPT_BIN{*x*}_REV        | Inter-packet times histogram *x*-th bin for the reverse direction                   |

On the [dataset metadata][metadata] page, packet histogram features are called `PHIST_SRC_SIZES`, `PHIST_DST_SIZES`, `PHIST_SRC_IPT`, `PHIST_DST_IPT`. Those are the names of database columns that are flattened to the _BIN{*x*} features.

## TCP features
Datasets with TLS over TCP traffic contain features indicating the presence of individual TCP flags in the flow.
Set `use_tcp_features` to use a subset of flags defined in `cesnet_datazoo.constants.SELECTED_TCP_FLAGS`.

| **Name**         | **Description**                                                                            |
|------------------|--------------------------------------------------------------------------------------------|
| FLAG_{*F*}       | Whether *F* flag was present in the forward (client to server) direction                   |
| FLAG_{*F*}_REV   | Whether *F* flag was present in the reverse (server to client) direction                   |

## Other fields
Datasets contain auxiliary information about samples, such as communicating hosts, flow times, and more fields extracted from the ClientHello message. The [dataset metadata][metadata] page lists available fields in individual datasets. 
Set `return_other_fields` to include those fields in returned dataframes. See [using dataloaders][using-dataloaders] for how other fields are handled in dataloaders.

| **Name**                | **Description**                                                                     |
|-------------------------|-------------------------------------------------------------------------------------|
| ID                      | Per-dataset unique flow identifier                                                  |
| TIME_FIRST              | Timestamp of the first packet                                                       |
| TIME_LAST               | Timestamp of the last packet                                                        |
| SRC_IP                  | Source IP address                                                                   |
| DST_IP                  | Destination IP address                                                              |
| DST_ASN                 | Destination Autonomous System number                                                |
| SRC_PORT                | Source port                                                                         |
| DST_PORT                | Destination port                                                                    |
| PROTOCOL                | Transport protocol                                                                  |
| TLS_SNI / QUIC_SNI      | Server Name Indication domain                                                       |
| TLS_JA3                 | JA3 fingerprint                                                                     |
| QUIC_VERSION            | QUIC protocol version                                                               |
| QUIC_USER_AGENT         | User agent string if available in the QUIC Initial Packet                           |
<!-- 
| APP                     | Web service label                                                                   |
| CATEGORY                | Service category label                                                              | 
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