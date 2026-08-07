# Execution DAG overview

The canonical DAG is `roadmap/tasks.yaml`; this file is only a human map.

```text
R00 -> R01
  \-> R02 -> D00 -> D01 -> D02 -> D04 --\
                    |      \-> D05 -------+-> B00/B01/B02/B03 -> B04
                    |       \-> D06 ------/                         |
                    \-> D03 -> D04                                  v
                                                               C00 -> C01/C02/C03 -> C04
                                                                                     |
                                                                                     v
                                                               M00 -> M01 -> M02 ----+--- S00
                                                                 |      |             |      
                                                                 |      +-> M04       +-> S02
                                                                 +-> M03 -> M05/M06 -> S01 -> S03
                                                                                         |
                                                                                     S04/S05 -> S06
                                                                                              |
                                                                                         P00 -> P01 -> P02
                                                                                          |      |       |
                                                                                          |      +-> A00 -> A01 -> A02 -> A03
                                                                                          |
                                                                                          +-> optimization O00..O07
                                                                                                   |
                                                                                                   +------> F00 -> F01 -> F02 -> F03
```

Optional OS tasks are not on the mandatory final path. Their activation is a late decision and destructive work is separately gated.
