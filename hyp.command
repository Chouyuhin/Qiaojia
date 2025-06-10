* This is a very simple hypoinverse test command file.
* It uses only a simple station and crust model file,
* with no station delay file or other options.
* Run hypoinverse, then type @test2000.hyp at the command prompt.

200 t 2000 0			/enable y2000 formats
H71 4 2 0			    /use new hypoinverse station format
DIS 4 100 5 3            /Main Distance weighting
RMS 4 0.5 1.5 6        /Residual weighting
ERR .1
POS 2.0
MIN 3                 /min number of stations
ZTR 8                  /trial depth
WET 1. .6 .3 .2       /weighting by pick quanlity
PRE 3, 3 0 0 9        /magnitude
* OUTPUT
ERF T
TOP F

STA 'stations_hypoinverse.dat'
LET 2 2 0                               /Net Sta Chn
TYP Read in crustal model(s):
CRH 1 'vel_model_P.crh'		/read crust model for Vp, here depth 0 is relative to the averge elevation of stations 
CRH 2 'vel_model_S.crh' 	/read crust model for Vs
SAL 1 2
PHS 'hypoInput.arc'		        /input phase file

FIL				        /automatically set phase format from file
ARC 'hypoOut.arc'		/output archive file
PRT 'prtOut.prt'		/output print file
SUM 'catOut.sum'        /output location summary
*RDM T
CAR 1
*LST 2
LOC				/locate the earthquake
STO
