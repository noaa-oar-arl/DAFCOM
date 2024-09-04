      MODULE PARAMS
      IMPLICIT NONE


      CHARACTER(4), DIMENSION(12,4), PARAMETER :: REPDATE = RESHAPE(
     &        [ "0109","0110","0114","0115",
     &          "0206","0207","0211","0212",
     &          "0306","0307","0311","0312",
     &          "0403","0404","0408","0409",
     &          "0508","0509","0513","0514",
     &          "0605","0606","0610","0611",
     &          "0710","0711","0715","0716",
     &          "0807","0808","0812","0813",
     &          "0911","0912","0916","0917",
     &          "1002","1003","1007","1008",
     &          "1106","1107","1111","1112",
     &          "1204","1205","1209","1210" ],
     &          [12,4], order = [2,1] )

      CHARACTER*16, DIMENSION(19), PARAMETER :: PM_INV  = 
     &       [ "PEC             ","PNO3            ","POC             ",
     &         "PSO4            ","PAL             ","PCA             ",
     &         "PCL             ","PFE             ","PH2O            ",
     &         "PK              ","PMG             ","PMN             ",
     &         "PMOTHR          ","PNA             ","PNH4            ",
     &         "PNCOM           ","PSI             ","PTI             ",
     &         "PMC             " ]

      CHARACTER*16, DIMENSION(27), PARAMETER :: VOC_INV  =
     &      [ "ACET            ","ACROLEIN        ","ALD2            ",
     &        "ALDX            ","BENZ            ","BUTADIENE13     ",
     &        "CH4             ","ETH             ","ETHA            ",
     &        "ETHY            ","ETOH            ","FORM            ",
     &        "IOLE            ","ISOP            ","KET             ",
     &        "MEOH            ","NAPH            ","OLE             ",
     &        "PAR             ","PRPA            ","FACD            ",
     &        "TERP            ","TOL             ","XYLMN           ",
     &        "AACD            ","APIN            ","SOAALK          " ]

      REAL,         DIMENSION(27), PARAMETER :: VOC_MW  =
     &      [ 58.1,  56.1,  44.,
     &        58.1,  78.1,  54.,
     &        16.,    28.,    30.1,
     &        26.,    46.1,  30.,
     &        56.1,  68.1,  72.1,
     &        32.,    128.2, 42.1,
     &        14.,    44.1,  46.,
     &        136.2, 92.1,  106.2,
     &        60.1,  136.2, 112. ]


      CHARACTER*16, DIMENSION(1), PARAMETER ::  CO_INV    = 
     &      [ "CO              " ]
      CHARACTER*16, DIMENSION(1), PARAMETER ::  NH3_INV   = 
     &      [ "NH3             " ]
      CHARACTER*16, DIMENSION(3), PARAMETER ::  NOX_INV   = 
     &      [ "NO              ","NO2             ","HONO            " ]
      REAL,         DIMENSION(3), PARAMETER ::  NOX_MW   =
     &      [ 30., 46., 47. ]
      CHARACTER*16, DIMENSION(1), PARAMETER ::  SO2_INV   = 
     &      [ "SO2             " ]


      CHARACTER*15, DIMENSION(9), PARAMETER :: SECTOR_NAM = 
     &               [ "afdust      ","ag          ","nonpt       ",
     &                 "nonroad     ","np_oilgas   ","rail        ",
     &                 "rwc         ","airports    ","onroad      " ]


      REAL,         DIMENSION(12), PARAMETER :: MDAY  =
     &               [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ]

      END MODULE PARAMS
