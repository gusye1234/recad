import os  
import sys
import subprocess
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

victim_model = ['mf', 'ncf', 'lightgcn']
white_attack_model = ['random','segment','bandwagon','average']
gray_attack_model = ['aush','aia','aushplus']
data = ['game', 'ml1m']
attack_model = white_attack_model + gray_attack_model

os.chdir(ROOT)
os.chdir('./examples')

header = ['attacker','victim','data','HR@10', 'HR@10_attack','HR@20', 'HR@20_attack','HR@50', 'HR@50_attack','HR@100', 'HR@100_attack']
firstline = ['---', '---', '---', 0.,0.,0.,0.,0.,0.,0.,0.]
dict = {header[i]:firstline[i] for i in range(len(header))}

df = pd.DataFrame(dict, index=[0])
df.to_csv('../mytest.csv', index=False)

# "target_id_list":[31,92]

epoch_list = [10]

for epoch in epoch_list:
   for dt in data:
      for vm in victim_model:  
         for am in attack_model:
            if am in white_attack_model:
               print(f"Running {vm}, {am} in {dt}(white attack) with {epoch}epochs")  
               subprocess.run(['python', 'from_command.py', '--attack_ratio=1', '--tqdm=0', '--filler_num=0', f'--data={dt}', f'--victim={vm}', f'--attack={am}', f'--rec_epoch={epoch}', f'--attack_epoch={epoch}'])
            else:
               print(f"Running {vm}, {am} in {dt}(gray attack) with {epoch}epochs")                 
               subprocess.run(['python', 'from_command.py', '--tqdm=0', '--filler_num=0', f'--data={dt}', f'--victim={vm}', f'--attack={am}', f'--rec_epoch={epoch}', f'--attack_epoch={epoch}'])
   df = pd.DataFrame([firstline])
   df.to_csv('../mytest.csv',mode='a',header=False, index=False)

os._exit(0)  
