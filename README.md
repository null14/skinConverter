# skinConverter

import skinConverter as sc
  
sc.MOVE_BONE = 10 # 1/1 scale  
#sc.MOVE_BONE = 1 # 1/10 scale  
sc.run(mesh='mesh',  tolerance = -1)  
  
  
skinCluster1 ->   
skinCluster2 ->   
------------------ skinCluster3.envelope 0  
------------------ wrap.envelope 1  
-------------------------------------------- -> convert -> skinCluster3.envelope 1  
sc.run(mesh='mesh',  tolerance = -1, skinHistory = True)
