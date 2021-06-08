#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3 -K 64 128 256 -L 1 -P 4 -H 100
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3 -K 64 128 256 -L 2 -P 4 -H 100
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3 -K 64 128 256 -L 3 -P 4 -H 100
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3 -K 64 128 256 -L 4 -P 4 -H 100
