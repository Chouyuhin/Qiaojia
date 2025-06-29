#  HYP	    HYPOINVERSE EARTHQUAKE LOCATION PROGRAM. version 1.40
hyp1.40  :  hyp.o hybda.o hycmd.o hysta.o hydel.o hyate.o hyfmc.o hycal.o \
hyxmc.o hycrh.o hycrt.o hystl.o hyopen.o hyinit.o hyphs.o hysou.o \
hytrl.o hyloc.o hysol.o hysvd.o hytra.o hytrh.o hytrt.o \
hymag.o hyrep.o hylst.o hysum.o hyinp.o hypro.o medwt.o utmcal.o \
hydelt.o hyedit.o hytime.o hybeg.o hypref.o hymagp.o linv.o linvol.o \
hyfile.o hylinv.o hylin.o hycre.o hytre.o trvcon.o hydatum.o allsubs.o
	gfortran -g hyp.o hybda.o hycmd.o hysta.o hydel.o hyate.o hyfmc.o hypref.o \
	hyxmc.o hycrh.o hycrt.o hystl.o hyopen.o hyinit.o hyphs.o hycal.o \
	hytrl.o hyloc.o hysol.o hysvd.o hytra.o hytrh.o hytrt.o \
	hymag.o hyrep.o hylst.o hysum.o hyinp.o hypro.o medwt.o utmcal.o \
	hydelt.o hyedit.o hytime.o hybeg.o linv.o linvol.o \
	hymagp.o hysou.o hyfile.o hylinv.o hylin.o hycre.o hytre.o trvcon.o \
	hydatum.o \
	-Bstatic allsubs.o -o hyp1.40
	cp hyp1.40 /c/mingw64/bin

# The -Bstatic option loads all fortran system subroutines into the executable
# so hyp can be copied to a similar system without fortran to run without
# the fortran system libraries (f77 compiler only?).

#  allsubs - collection of general subroutines into one file
#  This is the gfortran version of readq subroutine (no q format)

allsubs.o  :  allsubs.f 
	gfortran -c allsubs.f

#  HYP	   * MAIN HYPOINVERSE PROGRAM.
hyp.o  :  hyp.for common.inc
	gfortran -c hyp.for

#  HYBDA   * BLOCK DATA INITIALIZATION OF COMMON.
hybda.o  :  hybda.for common.inc
	gfortran -c hybda.for

#  HYBEG   * INITIALIZATION OF OTHER VARIABLES.
hybeg.o  :  hybeg.f common.inc
	gfortran -c hybeg.f

#  HYCMD   * GETS AND PROCESSES COMMANDS.
hycmd.o  :  hycmd.for common.inc
	gfortran -c hycmd.for

#  HYSTA   * READS IN STATIONS.
hysta.o  :  hysta.for common.inc
	gfortran -c hysta.for

#  HYDEL   * READS IN STATION DELAYS (FOR MULTIPLE MODELS).
hydel.o  :  hydel.for common.inc
	gfortran -c hydel.for

#  HYATE   * READS IN STATION ATTENUATION HISTORY.
hyate.o  :  hyate.for common.inc
	gfortran -c hyate.for

#  HYCAL   * READS IN STATION CAL FACTOR HISTORY.
hycal.o  :  hycal.for common.inc
	gfortran -c hycal.for

#  HYFMC   * READS IN STATION FMAG CORRECTIONS.
hyfmc.o  :  hyfmc.for common.inc
	gfortran -c hyfmc.for

#  HYFILE   * DETERMINES INPUT FILE TYPE.
hyfile.o  :  hyfile.for
	gfortran -c hyfile.for

#  HYXMC   * READS IN STATION XMAG CORRECTIONS.
hyxmc.o  :  hyxmc.for common.inc
	gfortran -c hyxmc.for

#  HYCRH   * READS IN HOMOGENOUS LAYER CRUSTAL MODELS.
hycrh.o  :  hycrh.for common.inc
	gfortran -c hycrh.for

#  HYCRT   * READS IN TRAVEL-TIME-TABLE CRUSTAL MODELS.
hycrt.o  :  hycrt.for common.inc
	gfortran -c hycrt.for

#  HYSTL   * OUTPUTS STATIONS, CRUST & PARAMETERS TO PRINT FILE.
hystl.o  :  hystl.for common.inc
	gfortran -c hystl.for

#  HYOPEN  * OPENS FILES FOR LOCATION RUN.
hyopen.o  :  hyopen.for common.inc
	gfortran -c hyopen.for

#  HYINIT  * INITIALIZES SOME VARIABLES FOR LOCATION RUN.
hyinit.o  :  hyinit.for common.inc
	gfortran -c hyinit.for

#  HYPHS   * READS IN PHASE DATA FOR ONE EVENT.
hyphs.o  :  hyphs.for common.inc
	gfortran -c hyphs.for

#  HYTRL   * SETS TRIAL HYPOCENTER.
hytrl.o  :  hytrl.for common.inc
	gfortran -c hytrl.for

#  HYLOC   * LOCATES ONE EVENT.
hyloc.o  :  hyloc.for common.inc
	gfortran -c hyloc.for

#  HYSOL   * PERFORMS INVERSION FOR ONE ITERATION.
hysol.o  :  hysol.for common.inc
	gfortran -c hysol.for

#  HYSVD     CANNED SINGLE-VALUE-DECOMPOSITION ROUTINE.
hysvd.o  :  hysvd.f
	gfortran -c hysvd.f

#  HYTRA   * MANAGE CRUST MODEL CHOICE & AVERAGING.
hytra.o  :  hytra.for common.inc
	gfortran -c hytra.for

#  HYTRH   * CALC TRAVEL TIMES AND DERIVS FOR HOMO LAYER MODEL.
hytrh.o  :  hytrh.for common.inc
	gfortran -c hytrh.for

#  HYTRT   * CALC TRAVEL TIMES AND DERIVS FROM TRAV-TIME TABLE.
hytrt.o  :  hytrt.for common.inc
	gfortran -c hytrt.for

#  HYLINV  * CALC TT AND DERIVS FROM LINEAR GRAD LAYER OVER HALFSPACE.
hylinv.o  :  hylinv.for common.inc
	gfortran -c hylinv.for

#  HYLIN  * CALC TT AND DERIVS FROM LINEAR GRAD MODEL.
hylin.o  :  hylin.for common.inc
	gfortran -c hylin.for

#  HYCRE  * READS IN HYPOELLIPSE LAYER CRUSTAL MODELS.
hycre.o  :  hycre.for common.inc
	gfortran -c hycre.for

#  HYTRE  * CALC TRAVEL TIMES AND DERIVS FOR HYPOELLIPSE LAYER MODEL.
hytre.o  :  hytre.for common.inc
	gfortran -c hytre.for

#  HYMAG   * COMPUTES DUR & MAX AMP MAGNITUDE.
hymag.o  :  hymag.for common.inc
	gfortran -c hymag.for

#  HYMAGP   * COMPUTES P AMP MAGNITUDE.
hymagp.o  :  hymagp.for common.inc
	gfortran -c hymagp.for

#  HYPREF  * SELECTS PREFERRED MAGNITUDE.
hypref.o  :  hypref.for common.inc
	gfortran -c hypref.for

#  HYREP   * REPORTS A LOCATION ON THE TERMINAL.
hyrep.o  :  hyrep.for common.inc
	gfortran -c hyrep.for

#  HYSOU   * TABULATES MOST COMMON DATA SOURCES.
hysou.o  :  hysou.for common.inc
	gfortran -c hysou.for

#  HYLST   * OUTPUTS DATA BY STATION TO PRINT & ARCHIVE FILES.
hylst.o  :  hylst.for common.inc
	gfortran -c hylst.for

#  HYSUM   * OUTPUTS SUMMARY RECORD (FINAL LOCATION).
hysum.o  :  hysum.for common.inc
	gfortran -c hysum.for

#  HYDATUM * COMPUTES DEPTH DATUM & CORRECTS DEPTH TO GEOID.
hydatum.o  :  hydatum.for common.inc
	gfortran -c hydatum.for

#  HYINP     FOR INTERACTIVE ENTRY OF PHASE DATA.
hyinp.o  :  hyinp.for
	gfortran -c hyinp.for

#  HYPRO   * INTERACTIVELY PROCESSES A SERIES OF EVENTS.
hypro.o  :  hypro.for common.inc
	gfortran -c hypro.for

#  MEDWT     COMPUTES THE WEIGHTED MEDIAN OF A SERIES OF MAGNITUDES.
medwt.o  :  medwt.for
	gfortran -c medwt.for

#  LINV     TRAVEL TIME & DERIVATIVES FOR HYPOELLIPSE GRADIENT MODEL.
linv.o  :  linv.for
	gfortran -c linv.for

#  LINVOL TRAVEL TIME & DERIVs FOR HYPOELLIPSE GRADIENT OVER HALPSPACE MODEL.
linvol.o  :  linvol.for
	gfortran -c linvol.for

#  TRVCON TRAVEL TIME & DERIVs FOR HYPOELLIPSE LAYER MODEL.
trvcon.o  :  trvcon.for
	gfortran -c trvcon.for

#  UTMCAL    CALCULATE STATION DISTANCES ON A UTM GRID
utmcal.o :  utmcal.for
	gfortran -c utmcal.for

#  INIT_EVENT  INITIALIZES CUSP FOR READING OF MEM DATA FILES (VAX ONLY)
#init_event.o  :  init_event.f
#	gfortran -c init_event.f

#  HYDELT DELETES FILES IN INTERACTIVE PROCESSING
hydelt.o  :  hydelt.f
	gfortran -c hydelt.f

#  HYEDIT  RUNS AN EDTIOR WITHIN A PROCESS
hyedit.o  :  hyedit.f
	gfortran -c hyedit.f

#  HYTIME    GETS CURRENT SYSTEM TIME FOR LABELING PRINT FILE
hytime.o  :  hytime.f
	gfortran -c hytime.f

#  ALSO USES THESE GENERAL PURPOSE SUBROUTINES FROM THE SUBROUTINE LIBRARY
#  KLAS      ASSIGNS A NAME AND NUMBER TO AN EVENT BASED ON LOCATION.
#  KLASS (NET 1), BOX2 (NET2) - USED BY KLAS.
#  ISITIN    DETERMINES IF EVENT IS IN A PARTICULAR REGION.
#  UPSTR     CONVERTS A STRING TO UPPER CASE.
#  JASK	     INTERACTIVE PROMPT & ENTRY OF AN INTEGER.
#  ASKC      INTERACTIVE PROMPT AND ENTRY OF A STRING.
#  ASKR      INTERACTIVE PROMPT AND ENTRY OF A REAL VALUE.
#  LASK      INTERACTIVE PROMPT AND ENTRY OF A LOGICAL VALUE.
#  LENG      DETERMINES THE NON-BLANK LENGTH OF A STRING.

#  Simple program to convert P delays to S delays using a constant POS
p2sdly  :  p2sdly.for
	gfortran -g p2sdly.for -Bstatic allsubs.o -o p2sdly
	cp p2sdly /home/calnet/klein/bin
