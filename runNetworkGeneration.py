import subprocess
import random
from pathlib import Path

# number of leaves
n_values = [50,100,500,1000]     
# number of reticulations    
r_ratio = [0.1,0.4]    
# max reticulation depth 
d_values = [1,5]          
# max label value
p_values = [1,3]          
# number of random seeds per configuration
num_runs = 25           

for n in n_values:
    for r in r_ratio:
        for d in d_values:
            for p in p_values:
                folder = Path(f"data/Networks_{n}_{int(r*n)}_{d}_{p}")
                folder.mkdir(exist_ok=True)
                for i in range(num_runs):
                    seed = random.randint(1, 1000000)
                    cmd = [
                        "python",
                        "genTreeChildNet.py",
                        "-n", str(n),
                        "-r", str(int(r*n)),
                        "-d", str(d),
                        "-p", str(p),
                        "-seed", str(seed),
                        "-f", f"data/Networks_{n}_{int(r*n)}_{d}_{p}"
                    ]
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, check=True)



