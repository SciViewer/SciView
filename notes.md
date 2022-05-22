## check unzip of unix vs synology

scidb overview 215 1 215
    Directory: 215 PDF count:  93187 Text count:  0
scidb dlunzip 215 215
scidb overview 215 1 215


# Split the unpaywall snapshot file into mutiple parts (each conataining 1M lines)

gunzip -c unpaywall_snapshot_2021-07-02T151134.jsonl.gz | split -l 1000000 - unpaywall_snapshot_2021-07-02T151134.jsonl.gz.part
    split: unpaywall_snapshot_2021-07-02T151134.jsonl.gz.partcw: Transport endpoint is not connected

gunzip -c unpaywall_snapshot_2021-07-02T151134.jsonl.gz | split -l 10000000 - unpaywall_snapshot_2021-07-02T151134.jsonl.gz.10MSplit


split -l 10000000 unpaywall_snapshot_2021-07-02T151134.jsonl unpay_2021_07_split_


ssh fishsalat@192.168.0.18 -p22
cd volume1/SciMagDir/Reference_Databases/unpaywall/
gunzip -c unpaywall_snapshot_2021-07-02T151134.jsonl.gz | split -l 10000000 - unpaywall_snapshot_2021-07-02T151134.jsonl.gz.10Msplit
split -l 1000000 unpaywall_snapshot_2021-07-02T151134.jsonl