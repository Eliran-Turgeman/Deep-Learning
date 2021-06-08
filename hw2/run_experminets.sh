#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 2 -P 2 -H 100
#srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 4 -P 2 -H 100
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 8 -P 4 -H 100
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 32 -L 16 -P 4 -H 100
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 2 -P 4 -H 100
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 4 -P 4 -H 100
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 8 -P 4 -H 100
srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n test -K 64 -L 16 -P 4 -H 100
