import datetime
from art import text2art
import os

__description__ = 'A Multi-Objective Bayesian global optimization package'
__author__ = 'Bin Cao, Advanced Materials Thrust, Hong Kong University of Science and Technology (Guangzhou)'
__author_email__ = 'binjacobcao@gmail.com'
__url__ = 'https://github.com/Bin-Cao/MultiBgolearn'


os.makedirs('MultiBgolearn', exist_ok=True)
now = datetime.datetime.now()
formatted_date_time = now.strftime('%Y-%m-%d %H:%M:%S')
print(text2art("Multi-Objs"))
print('Multi-Bgolearn, Bin CAO, HKUST(GZ)' )
print('URL : https://github.com/Bin-Cao/MultiBgolearn')
print('Executed on :',formatted_date_time, ' | Have a great day.')  
print('='*80)