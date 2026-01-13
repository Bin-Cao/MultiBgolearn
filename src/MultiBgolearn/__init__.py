import datetime
from art import text2art
import os

__description__ = 'A Multi-Objective Bayesian global optimization package'
__author__ = 'Bin Cao, Advanced Materials Thrust, Hong Kong University of Science and Technology (Guangzhou)'
__author_email__ = 'binjacobcao@gmail.com'
__url__ = 'https://github.com/Bin-Cao/MultiBgolearn'
__paper__ = 'https://doi.org/10.48550/arXiv.2601.06820'
__Doc__ = 'https://bgolearn.netlify.app/'

os.makedirs('MultiBgolearn', exist_ok=True)
now = datetime.datetime.now()
formatted_date_time = now.strftime('%Y-%m-%d %H:%M:%S')
print('MultiBgolearn, Bin CAO, HKUST(GZ)' )
print('Paper : https://doi.org/10.48550/arXiv.2601.06820')
print('URL : https://github.com/Bin-Cao/MultiBgolearn')
print('Doc : https://bgolearn.netlify.app/')
print('Executed on :',formatted_date_time, ' | Have a great day.')  
print('='*80)