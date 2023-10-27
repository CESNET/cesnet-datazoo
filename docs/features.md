# Features

## Details about packet histograms and packet sequences
Due to differences in implementation between packet sequences ([pstats.cpp](https://github.com/CESNET/ipfixprobe/blob/master/process/pstats.cpp)) and packet histogram ([phist.cpp](https://github.com/CESNET/ipfixprobe/blob/master/process/phists.cpp)) plugins of the ipfixprobe exporter, the number of packets in those features can differ. The differences are summarized in the following table.
Note that this is related to TLS over TCP datasets. For QUIC, there is no detection of retransmissions or out-of-order packets. Also, QUIC acknowledgment packets are included in both packet sequences and packet histograms.

| *TLS over TCP datasets*                                       | Packet histograms | PPI sequence  | PACKETS and PACKET_REV |
|---------------------------------------------------------------|-------------------|---------------|------------------------|
| **Zero-length packets**<br/>(without L4 payload, e.g. ACKs)   | Not included      | Not included  | Included               |
| **Retransmissions**<br/>(including out-of-order packets)      | Included          | Not included* | Included               |

**For PPI sequences, the implementation for the detection of retransmissions and out-of-order packets is not perfect. Packets with a non-increasing SEQ number are skipped.*
