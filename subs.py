#
# Leer datos de T/P, graficar
#

def read_t_p_data() :
   import numpy as np


   # Nombres de archivos
   Tfile    = 'MON_T_CRU_19012015.csv'
   Pfile    = 'MON_P_CRU_19012015.csv'
   sta_file = 'BASIN_CHARACTERISTICS.csv'

   # Read data files
   basin = np.loadtxt(sta_file,skiprows=1,delimiter=',')
   Temp  = np.loadtxt(Tfile,   skiprows=1,delimiter=',')
   Prec  = np.loadtxt(Pfile,   skiprows=1,delimiter=',')

   # Organize files
   T = Temp[:,1:]
   P = Prec[:,1:]

   sta  = basin[:,0]
   lon  = basin[:,1]
   lat  = basin[:,2]
   area = basin[:,3]
   elev = basin[:,4]

   return T,P,sta,lon,lat


