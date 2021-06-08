#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp2 -K 32 64 128 -L 3 -P 2 -H 100 -M ycn
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp2 -K 32 64 128 -L 6 -P 4 -H 100 -M ycn
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp2 -K 32 64 128 -L 9 -P 6 -H 100 -M ycn
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp2 -K 32 64 128 -L 12 -P 8 -H 100 -M ycn
